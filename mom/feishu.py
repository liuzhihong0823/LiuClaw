from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from .store import MomStore
from .types import ChatAttachment, ChatContext, ChatEvent, ChatInfo, ChatUser

_FEISHU_TEXT_LIMIT = 4000


def _strip_mentions(text: str, mentions: list[str]) -> str:
    """移除消息文本中的 @ 提及内容，保留真正的用户意图。"""
    normalized = text
    for mention in mentions:
        normalized = normalized.replace(mention, "")
    return " ".join(normalized.split())


def _normalize_text_message(text: str) -> str:
    """规范化飞书文本消息，过滤非法字符并控制长度。"""
    cleaned = "".join(ch for ch in str(text) if ch in "\n\t" or ord(ch) >= 32)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        cleaned = "(empty message)"
    if len(cleaned) > _FEISHU_TEXT_LIMIT:
        cleaned = cleaned[: _FEISHU_TEXT_LIMIT - 16].rstrip() + "\n...(truncated)"
    return cleaned


@dataclass(slots=True)
class FeishuConfig:
    """飞书接入配置，包含鉴权参数和服务监听方式。"""

    app_id: str  # 飞书应用 App ID。
    app_secret: str  # 飞书应用密钥。
    connection_mode: str = "long_connection"  # 接入模式。
    verification_token: str = ""  # webhook 校验 token。
    encrypt_key: str = ""  # webhook/长连接加密 key。
    bind_host: str = "127.0.0.1"  # webhook 监听地址。
    bind_port: int = 8123  # webhook 监听端口。
    base_url: str = "https://open.feishu.cn/open-apis"  # 飞书 OpenAPI 基础地址。


class FeishuBotTransport:
    def __init__(self, handler, config: FeishuConfig, *, client: httpx.AsyncClient | None = None) -> None:
        """初始化飞书传输层，负责消息收发、事件解析和上下文封装。"""
        self.handler = handler
        self.config = config
        self.client = client or httpx.AsyncClient(timeout=20.0)
        self._tenant_access_token: str | None = None
        self._users: dict[str, ChatUser] = {}
        self._chats: dict[str, ChatInfo] = {}

    async def close(self) -> None:
        """关闭底层 HTTP 客户端连接。"""
        await self.client.aclose()

    async def get_tenant_access_token(self) -> str:
        """获取并缓存飞书 tenant_access_token。"""
        if self._tenant_access_token:
            return self._tenant_access_token
        response = await self.client.post(
            f"{self.config.base_url}/auth/v3/tenant_access_token/internal",
            json={"app_id": self.config.app_id, "app_secret": self.config.app_secret},
        )
        response.raise_for_status()
        payload = response.json()
        self._tenant_access_token = str(payload["tenant_access_token"])
        return self._tenant_access_token

    async def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        """统一封装带鉴权的飞书 OpenAPI 请求。"""
        headers = dict(kwargs.pop("headers", {}))
        headers["Authorization"] = f"Bearer {await self.get_tenant_access_token()}"
        response = await self.client.request(method, f"{self.config.base_url}{path}", headers=headers, **kwargs)
        if response.is_error:
            detail = response.text.strip()
            raise RuntimeError(
                f"Feishu API {method} {path} failed with {response.status_code}: {detail or response.reason_phrase}"
            )
        if not response.content:
            return {}
        return response.json()

    async def _create_message(self, receive_id: str, receive_id_type: str, payload: dict[str, Any]) -> str | None:
        """调用飞书消息发送接口创建一条消息。"""
        data = await self._request(
            "POST",
            f"/im/v1/messages?receive_id_type={receive_id_type}",
            json={"receive_id": receive_id, **payload},
        )
        return data.get("data", {}).get("message_id")

    async def send_text(
        self,
        chat_id: str,
        text: str,
        *,
        reply_to: str | None = None,
        open_id: str | None = None,
        is_direct: bool = False,
    ) -> str | None:
        """发送文本消息；优先走回复，失败后回退到普通发消息。"""
        payload = {"msg_type": "text", "content": json.dumps({"text": _normalize_text_message(text)}, ensure_ascii=False)}
        reply_error: Exception | None = None
        if reply_to:
            try:
                data = await self._request("POST", f"/im/v1/messages/{reply_to}/reply", json=payload)
                return data.get("data", {}).get("message_id")
            except Exception as exc:
                reply_error = exc
        try:
            return await self._create_message(chat_id, "chat_id", payload)
        except Exception:
            if is_direct and open_id:
                return await self._create_message(open_id, "open_id", payload)
            if reply_error is not None:
                raise reply_error
            raise

    async def update_text(self, message_id: str, text: str) -> str | None:
        """更新已发送消息的正文，用于流式输出或占位消息替换。"""
        try:
            await self._request("PATCH", f"/im/v1/messages/{message_id}", json={"content": json.dumps({"text": text}, ensure_ascii=False)})
            return message_id
        except Exception:
            return None

    async def send_file(
        self,
        chat_id: str,
        file_path: str,
        title: str | None = None,
        *,
        open_id: str | None = None,
        is_direct: bool = False,
    ) -> str | None:
        """上传本地文件到飞书，并把文件消息发送到目标会话。"""
        path = Path(file_path)
        with path.open("rb") as handle:
            data = await self._request("POST", "/im/v1/files", files={"file": (title or path.name, handle)})
        file_key = data.get("data", {}).get("file_key")
        if not file_key:
            return None
        payload = {
            "msg_type": "file",
            "content": json.dumps({"file_key": file_key}, ensure_ascii=False),
        }
        try:
            return await self._create_message(chat_id, "chat_id", payload)
        except Exception:
            if is_direct and open_id:
                return await self._create_message(open_id, "open_id", payload)
            raise

    async def download_attachment(self, attachment: ChatAttachment) -> bytes:
        """下载飞书消息中的附件二进制内容。"""
        file_key = attachment.file_key or attachment.metadata.get("file_key")
        if not file_key:
            raise ValueError("attachment file_key is required")
        token = await self.get_tenant_access_token()
        response = await self.client.get(
            f"{self.config.base_url}/im/v1/messages/{attachment.message_id}/resources/{file_key}",
            headers={"Authorization": f"Bearer {token}"},
            params={"type": "file"},
        )
        response.raise_for_status()
        return response.content

    def create_context(self, event: ChatEvent, store: MomStore) -> ChatContext:
        """把飞书事件包装成运行期 ChatContext，供 runner 统一调用。"""
        response_holder = {"main": None}

        async def respond(text: str, log: bool = True) -> str | None:
            if response_holder["main"] is None:
                response_holder["main"] = await self.send_text(
                    event.chat_id,
                    text,
                    reply_to=event.message_id,
                    open_id=event.sender_id,
                    is_direct=event.is_direct,
                )
            else:
                updated = await self.update_text(str(response_holder["main"]), text)
                if updated is None:
                    response_holder["main"] = await self.send_text(
                        event.chat_id,
                        text,
                        reply_to=event.message_id,
                        open_id=event.sender_id,
                        is_direct=event.is_direct,
                    )
            return response_holder["main"]

        async def replace_message(text: str) -> str | None:
            return await respond(text, False)

        async def respond_detail(text: str) -> str | None:
            anchor = response_holder["main"] or event.message_id
            return await self.send_text(
                event.chat_id,
                text,
                reply_to=str(anchor),
                open_id=event.sender_id,
                is_direct=event.is_direct,
            )

        async def upload_file(file_path: str, title: str | None = None) -> str | None:
            return await self.send_file(
                event.chat_id,
                file_path,
                title,
                open_id=event.sender_id,
                is_direct=event.is_direct,
            )

        async def set_working(_: bool) -> None:
            return None

        async def delete_message() -> None:
            return None

        return ChatContext(
            message=event,
            chat_name=event.chat_name,
            users=list(self._users.values()),
            chats=list(self._chats.values()),
            respond=respond,
            replace_message=replace_message,
            respond_detail=respond_detail,
            upload_file=upload_file,
            set_working=set_working,
            delete_message=delete_message,
        )

    async def ingest_callback(self, payload: dict[str, Any], store: MomStore) -> dict[str, Any]:
        """处理 webhook 模式的回调请求，并写入本地事件日志。"""
        if payload.get("type") == "url_verification":
            return {"challenge": payload.get("challenge", "")}
        token = payload.get("token") or payload.get("header", {}).get("token", "")
        if self.config.verification_token and token and token != self.config.verification_token:
            return {"code": 403, "msg": "invalid token"}
        event = self.parse_event(payload)
        if event is None:
            return {"code": 0}
        store.log_event(event)
        await self.handler.handle_chat_event(event)
        return {"code": 0}

    async def ingest_long_connection_event(self, payload: dict[str, Any], store: MomStore) -> None:
        """处理长连接模式收到的事件，并立即分发给应用层。"""
        event = self.parse_event(payload)
        if event is None:
            return
        store.log_event(event)
        await self.handler.handle_chat_event(event)

    def parse_event(self, payload: dict[str, Any]) -> ChatEvent | None:
        """把飞书原始事件解析为内部统一的 ChatEvent。"""
        event = payload.get("event") or payload.get("data", {}).get("event") or payload.get("data") or {}
        message = event.get("message") or {}
        sender = event.get("sender") or {}
        if not message or not sender:
            return None
        chat_id = str(message.get("chat_id") or "")
        message_id = str(message.get("message_id") or "")
        sender_id = str(((sender.get("sender_id") or {}).get("open_id")) or sender.get("id") or "")
        sender_name = str((sender.get("sender_name")) or (sender.get("name")) or sender_id or "unknown")
        chat_type = str(message.get("chat_type") or "group")
        mentions = [item.get("name", "") for item in (event.get("mentions") or [])]
        text = ""
        content_raw = message.get("content")
        if isinstance(content_raw, str):
            try:
                content = json.loads(content_raw)
            except json.JSONDecodeError:
                content = {"text": content_raw}
        else:
            content = content_raw or {}
        if isinstance(content, dict):
            text = str(content.get("text") or "")
        attachments: list[ChatAttachment] = []
        for item in content.get("files", []) if isinstance(content, dict) else []:
            attachments.append(
                ChatAttachment(
                    original_name=str(item.get("file_name") or "attachment"),
                    file_key=item.get("file_key"),
                )
            )
        normalized_text = _strip_mentions(text, mentions)
        is_direct = chat_type == "p2p"
        is_trigger = is_direct or bool(mentions)
        event_obj = ChatEvent(
            platform="feishu",
            chat_id=chat_id,
            message_id=message_id or uuid4().hex,
            sender_id=sender_id,
            sender_name=sender_name,
            text=normalized_text,
            attachments=attachments,
            is_direct=is_direct,
            is_trigger=is_trigger,
            occurred_at=datetime.now(timezone.utc),
            chat_name=event.get("chat_name"),
            mentions=mentions,
            metadata={"raw_event": event},
        )
        self._users[sender_id] = ChatUser(id=sender_id, name=sender_name, display_name=sender_name)
        if chat_id:
            self._chats[chat_id] = ChatInfo(id=chat_id, name=str(event.get("chat_name") or chat_id))
        return event_obj

    async def _serve_webhook(self, store: MomStore) -> None:
        """以本地 HTTP 服务方式接收飞书 webhook 回调。"""
        loop = asyncio.get_running_loop()
        transport = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                future = asyncio.run_coroutine_threadsafe(transport.ingest_callback(payload, store), loop)
                body = future.result()
                raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, format: str, *args) -> None:  # noqa: A003
                return None

        server = ThreadingHTTPServer((self.config.bind_host, self.config.bind_port), Handler)
        await loop.run_in_executor(None, server.serve_forever)

    def _marshal_lark_payload(self, raw_event: Any, lark_module: Any) -> dict[str, Any] | None:
        """把长连接 SDK 提供的事件对象转成标准字典结构。"""
        if isinstance(raw_event, dict):
            return raw_event
        payload_text = lark_module.JSON.marshal(raw_event)
        payload = json.loads(payload_text)
        if not isinstance(payload, dict):
            return None
        if "event" in payload:
            return payload
        return {"event": payload}

    def _build_long_connection_client(self, store: MomStore, loop: asyncio.AbstractEventLoop) -> Any:
        """构造飞书长连接客户端，并绑定消息事件回调。"""
        try:
            import lark_oapi as lark
        except ImportError as exc:
            raise RuntimeError(
                "Long connection mode requires `lark-oapi`. Install it first: pip install lark-oapi"
            ) from exc

        def on_im_message_receive(data: Any) -> None:
            payload = self._marshal_lark_payload(data, lark)
            if payload is None:
                return
            future = asyncio.run_coroutine_threadsafe(self.ingest_long_connection_event(payload, store), loop)
            future.result()

        event_handler = (
            lark.EventDispatcherHandler.builder(self.config.encrypt_key, self.config.verification_token)
            .register_p2_im_message_receive_v1(on_im_message_receive)
            .build()
        )
        return lark.ws.Client(self.config.app_id, self.config.app_secret, event_handler=event_handler)

    async def _serve_long_connection(self, store: MomStore) -> None:
        """以长连接模式接入飞书事件流。"""
        loop = asyncio.get_running_loop()

        # lark_oapi.ws.Client captures an event loop during construction and then
        # drives it with run_until_complete() inside start(). Build and start the
        # client inside a worker thread so it owns a non-running loop.
        def run_client() -> None:
            client = self._build_long_connection_client(store, loop)
            client.start()

        await asyncio.to_thread(run_client)

    async def serve(self, store: MomStore) -> None:
        """根据配置选择 webhook 或长连接模式启动飞书服务。"""
        mode = self.config.connection_mode.strip().lower()
        if mode in {"webhook", "http"}:
            await self._serve_webhook(store)
            return
        await self._serve_long_connection(store)
