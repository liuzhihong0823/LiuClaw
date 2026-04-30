from __future__ import annotations

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import Processor, Transformation
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.styles import Style

from .state import InteractiveState


class TranscriptStyleProcessor(Processor):
    """根据 transcript 行号为主输出区应用样式。"""

    def __init__(self, state: InteractiveState) -> None:
        """保存状态引用，便于按行读取样式信息。"""

        self.state = state  # 当前交互状态。

    def apply_transformation(self, transformation_input):
        """为当前行附加统一的样式类。"""

        style_name = self.state.transcript_line_styles.get(transformation_input.lineno)
        if not style_name:
            return Transformation(transformation_input.fragments)
        fragments = []
        for fragment in transformation_input.fragments:
            if len(fragment) == 2:
                fragment_style, text = fragment
                combined = f"{fragment_style} class:{style_name}".strip()
                fragments.append((combined, text))
            else:
                fragment_style, text, handler = fragment
                combined = f"{fragment_style} class:{style_name}".strip()
                fragments.append((combined, text, handler))
        return Transformation(fragments)


class MainOutputBufferControl(BufferControl):
    """主输出区专用 BufferControl，显式接管滚轮事件。"""

    def __init__(self, renderer: "InteractiveRenderer", *args, **kwargs) -> None:
        """保存 renderer 引用，便于把滚轮事件路由到统一滚动接口。"""

        super().__init__(*args, **kwargs)
        self.renderer = renderer  # 所属渲染器。

    def mouse_handler(self, mouse_event):
        """优先处理主输出区滚轮事件，其他事件沿用 BufferControl 默认行为。"""

        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self.renderer.handle_main_mouse_scroll("up")
            return None
        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self.renderer.handle_main_mouse_scroll("down")
            return None
        if self.focus_on_click() and mouse_event.event_type == MouseEventType.MOUSE_UP:
            get_app().layout.current_control = self
            return None
        return super().mouse_handler(mouse_event)


class InteractiveRenderer:
    """负责把交互状态渲染成 prompt_toolkit 界面。"""

    def __init__(self, state: InteractiveState) -> None:
        """初始化渲染器并保存状态对象。"""

        self.state = state  # 当前交互状态。
        self.application: Application | None = None  # prompt_toolkit Application 实例。
        self.input_buffer = Buffer(multiline=True)  # 用户输入缓冲区。
        self.transcript_buffer = Buffer(read_only=True, document=Document(state.transcript_text, cursor_position=len(state.transcript_text)))  # 主输出只读缓冲区。
        self.main_control = MainOutputBufferControl(
            self,
            buffer=self.transcript_buffer,
            focusable=True,
            focus_on_click=True,
            include_default_input_processors=False,
            input_processors=[TranscriptStyleProcessor(state)],
        )
        self.main_window = Window(self.main_control, wrap_lines=True, always_hide_cursor=True)  # 主输出窗口。
        self.sidebar_window = Window(FormattedTextControl(self._render_sidebar), width=Dimension(preferred=36), wrap_lines=True)  # 侧边栏窗口。
        self.input_window = Window(BufferControl(buffer=self.input_buffer), height=Dimension(min=3, preferred=5), wrap_lines=True)  # 输入窗口。
        self._pending_follow_after_render = False  # 下次渲染后是否需要跟随到底部。
        self._last_main_height = 0  # 上次记录的主窗口高度。

    def build_application(self, controller) -> Application:
        """创建完整的 prompt_toolkit Application。"""

        key_bindings = self._build_key_bindings(controller)
        body = VSplit(
            [
                self.main_window,
                Window(width=1, char="|"),
                self.sidebar_window,
            ]
        )
        root = HSplit(
            [
                body,
                Window(height=1, char="-"),
                Window(FormattedTextControl(self._render_input_hint), height=1),
                self.input_window,
                Window(height=1, char="-"),
                Window(FormattedTextControl(self._render_status_bar), height=1),
            ]
        )
        application = Application(
            layout=Layout(root, focused_element=self.input_window),
            key_bindings=key_bindings,
            full_screen=True,
            mouse_support=True,
            style=self._build_style(),
            refresh_interval=0.1,
            after_render=lambda app: self.update_scroll_after_render(),
        )
        self.application = application
        return application

    def invalidate(self) -> None:
        """请求界面刷新。"""

        if self.application is not None:
            self.application.invalidate()

    def refresh_style(self) -> None:
        """在主题切换后立即刷新应用样式。"""

        if self.application is not None:
            self.application.style = self._build_style()
            self.invalidate()

    def _build_style(self) -> Style:
        """根据当前主题生成界面样式。"""

        return Style.from_dict(dict(self.state.theme_styles))

    def _build_key_bindings(self, controller) -> KeyBindings:
        """构造交互层使用的快捷键集合。"""

        kb = KeyBindings()

        @kb.add("enter", filter=Condition(lambda: self.state.submit_on_enter and not self.state.is_running))
        def _(event) -> None:
            controller.submit_current_buffer()

        @kb.add("escape", "enter")
        def _(event) -> None:
            self.input_buffer.insert_text("\n")

        @kb.add("escape", filter=Condition(lambda: not self.focused_on_input()))
        def _(event) -> None:
            controller.focus_input()

        @kb.add("c-c")
        def _(event) -> None:
            controller.cancel_current()

        @kb.add("c-l")
        def _(event) -> None:
            controller.clear_output()

        @kb.add("tab")
        def _(event) -> None:
            controller.autocomplete_buffer()

        @kb.add("c-r")
        def _(event) -> None:
            controller.show_help("历史搜索可使用上下方向键浏览输入历史。")

        @kb.add("pageup")
        def _(event) -> None:
            controller.scroll_main_page_up()

        @kb.add("pagedown")
        def _(event) -> None:
            controller.scroll_main_page_down()

        @kb.add("end")
        def _(event) -> None:
            controller.jump_to_latest()

        @kb.add("home")
        def _(event) -> None:
            controller.jump_to_oldest()

        @kb.add("escape", "up")
        def _(event) -> None:
            controller.scroll_main_up()

        @kb.add("escape", "down")
        def _(event) -> None:
            controller.scroll_main_down()

        @kb.add("escape", "f")
        def _(event) -> None:
            controller.jump_to_latest()

        @kb.add("f6")
        def _(event) -> None:
            controller.toggle_focus()

        return kb

    def _render_sidebar(self):
        """渲染侧边状态栏。"""

        scroll_mode = "Follow Latest" if self.state.auto_follow_output else "Viewing History"
        lines = [
            ("class:status", f"Session: {self.state.session_id}\n"),
            ("class:status", f"Session File: {self.state.session_file or '-'}\n"),
            ("class:status", f"Leaf: {self.state.leaf_id or '-'}\n"),
            ("class:status", f"Model: {self.state.model_id}\n"),
            ("class:status", f"Thinking: {self.state.thinking or 'default'}\n"),
            ("class:status", f"CWD: {self.state.cwd}\n"),
            ("class:status", f"Running: {'yes' if self.state.is_running else 'no'}\n"),
            ("class:status", f"Theme: {self.state.theme}\n"),
            ("class:status", f"Current Tool: {self.state.current_tool or '-'}\n"),
            ("class:status", f"Scroll: {scroll_mode}\n"),
            ("class:status", f"New Output: {self.state.unseen_output_updates}\n"),
            ("class:status", "\nRecent Sessions:\n"),
        ]
        for item in self.state.recent_sessions[:5]:
            lines.append(("class:status", f"- {item.get('session_id')} {item.get('title', '')}\n"))
        lines.append(("class:status", "\nStatus Timeline:\n"))
        for item in self.state.status_timeline[-8:]:
            lines.append(("class:status", f"- {item}\n"))
        return lines

    def _render_input_hint(self):
        """渲染输入区说明。"""

        mode = "Enter send / Alt-Enter newline" if self.state.submit_on_enter else "Alt-Enter send / Enter newline"
        return [("class:input_prompt", f"Input: {mode}")]

    def _render_status_bar(self):
        """渲染底部状态栏。"""

        error = f" | Error: {self.state.last_error}" if self.state.last_error else ""
        running = "RUNNING" if self.state.is_running else "IDLE"
        follow = "Follow Latest" if self.state.auto_follow_output else "Viewing History"
        pending = f" | New output: {self.state.unseen_output_updates} | End jump latest" if self.state.unseen_output_updates else " | End jump latest"
        focus = "Input Focus" if self.focused_on_input() else "Main Focus"
        return [
            (
                "class:status_bar",
                f" Ctrl-C cancel | Ctrl-L clear | Mouse wheel scroll output | PgUp/PgDn scroll | Home jump oldest | End jump latest | Esc back to input | Esc+Up/Down line scroll | F6 focus | {running} | {follow} | {focus} | Top: {self.state.main_view_top_display_line}{pending}{error}",
            )
        ]

    def scroll_main_lines(self, delta: int) -> None:
        """按行滚动主输出区。"""

        if delta == 0:
            return
        current = self.main_window.vertical_scroll
        max_scroll = self._max_scroll()
        self.main_window.vertical_scroll = min(max(0, current + delta), max_scroll)
        self._update_view_mode_from_viewport()
        self.invalidate()

    def scroll_main_pages(self, delta_pages: int) -> None:
        """按页滚动主输出区。"""

        page = max(1, self.get_main_viewport()["window_height"] - 1)
        self.scroll_main_lines(delta_pages * page)

    def scroll_main_to_top(self, invalidate: bool = True) -> bool:
        """跳转到最早可见位置。"""

        changed = self.main_window.vertical_scroll != 0
        self.main_window.vertical_scroll = 0
        self.state.mark_history_view()
        self.state.main_view_top_display_line = 0
        if invalidate:
            self.invalidate()
        return changed

    def scroll_main_to_bottom(self, *, mark_mode: str = "jumped", invalidate: bool = True) -> bool:
        """直接跳转到主输出区底部。"""

        if self.state.last_rendered_revision != self.state.transcript_revision:
            self.sync_transcript_content()
        target = self._bottom_scroll_target()
        if target is None:
            self.state.main_pending_jump_to_bottom = True
            self._pending_follow_after_render = True
            if mark_mode == "jumped":
                self.state.mark_jumped_to_latest()
            else:
                self.state.mark_latest_view()
            return False
        changed = self.main_window.vertical_scroll != target
        self.main_window.vertical_scroll = target
        self._pending_follow_after_render = False
        self.state.main_pending_jump_to_bottom = False
        if mark_mode == "jumped":
            self.state.mark_jumped_to_latest()
        else:
            self.state.mark_latest_view()
        self.state.main_view_top_display_line = target
        if invalidate:
            self.invalidate()
        return changed

    def is_main_at_bottom(self) -> bool:
        """判断主输出区当前是否已经接近底部。"""

        viewport = self.get_main_viewport()
        max_scroll = max(0, viewport["content_height"] - viewport["window_height"])
        if max_scroll <= 0:
            return True
        return self.main_window.vertical_scroll >= max_scroll - 1

    def follow_output_if_needed(self) -> None:
        """在允许自动跟随时请求重绘后移动到最新消息。"""

        if self.state.auto_follow_output:
            self._pending_follow_after_render = True

    def update_scroll_after_render(self) -> None:
        """在界面重绘后重新修正主输出区滚动位置，并处理窗口尺寸变化。"""

        viewport = self.get_main_viewport()
        current_height = viewport["window_height"]
        resized = current_height != self._last_main_height
        if resized:
            self._last_main_height = current_height
        self.state.main_last_rendered_content_height = viewport["content_height"]
        if self.state.auto_follow_output and (self._pending_follow_after_render or resized or self.state.main_pending_jump_to_bottom):
            changed = self.scroll_main_to_bottom(mark_mode="latest", invalidate=False)
            if changed:
                self.invalidate()
            return
        max_scroll = max(0, viewport["content_height"] - viewport["window_height"])
        if self.main_window.vertical_scroll > max_scroll:
            self.main_window.vertical_scroll = max_scroll
        self.state.main_view_top_display_line = self.main_window.vertical_scroll
        self._update_view_mode_from_viewport()

    def sync_transcript_content(self, state: InteractiveState | None = None) -> bool:
        """只同步 transcript 内容，不直接调整视口位置。"""

        current_state = state or self.state
        if current_state.transcript_revision == current_state.last_rendered_revision:
            return False
        text = current_state.transcript_text
        cursor_position = min(self.transcript_buffer.cursor_position, len(text))
        self.transcript_buffer.set_document(Document(text, cursor_position=cursor_position), bypass_readonly=True)
        current_state.last_rendered_revision = current_state.transcript_revision
        return True

    def reconcile_viewport_after_content_change(self) -> None:
        """在 transcript 内容变化后根据最新/历史模式协调视口。"""

        if self.state.main_view_mode == "latest":
            self.state.main_pending_jump_to_bottom = True
            self._pending_follow_after_render = True
            return
        self.state.main_pending_jump_to_bottom = False
        self.state.mark_history_view()

    def get_main_viewport(self) -> dict[str, int]:
        """读取主输出区当前视口信息。"""

        render_info = self.main_window.render_info
        if render_info is None:
            content_height = max(self.state.transcript_line_count, self.state.main_last_rendered_content_height)
            window_height = self._window_height(self.main_window)
            return {
                "content_height": content_height,
                "window_height": window_height,
                "top": self.main_window.vertical_scroll,
                "bottom": min(content_height, self.main_window.vertical_scroll + window_height),
            }
        content_height = max(getattr(render_info, "content_height", self.state.transcript_line_count), self.state.transcript_line_count)
        vertical_scroll = getattr(render_info, "vertical_scroll", self.main_window.vertical_scroll)
        window_height = getattr(render_info, "window_height", self._window_height(self.main_window))
        return {
            "content_height": content_height,
            "window_height": window_height,
            "top": vertical_scroll,
            "bottom": vertical_scroll + window_height,
        }

    def _bottom_scroll_target(self) -> int | None:
        """根据当前 transcript 文档和窗口高度计算底部滚动位置。"""

        viewport = self.get_main_viewport()
        window_height = viewport["window_height"]
        if window_height <= 0:
            return None
        return max(0, viewport["content_height"] - window_height)

    def _max_scroll(self) -> int:
        """计算主输出区可滚动的最大行偏移。"""

        target = self._bottom_scroll_target()
        if target is None:
            return 0
        return target

    def focus_main(self) -> None:
        """把焦点切换到主输出区。"""

        if self.application is not None:
            self.application.layout.focus(self.main_window)

    def focus_input(self) -> None:
        """把焦点切换回输入区。"""

        if self.application is not None:
            self.application.layout.focus(self.input_window)

    def focus_input_if_idle(self) -> None:
        """在空闲状态下自动把焦点切回输入区。"""

        if not self.state.is_running:
            self.focus_input()

    def handle_main_mouse_scroll(self, direction: str) -> None:
        """显式处理主输出区的鼠标滚轮事件。"""

        if direction == "up":
            self.scroll_main_lines(-3)
            return
        if direction == "down":
            self.scroll_main_lines(3)

    def focused_on_input(self) -> bool:
        """判断当前焦点是否位于输入区。"""

        if self.application is None:
            return True
        return self.application.layout.current_window is self.input_window

    @staticmethod
    def _window_height(window: Window) -> int:
        """读取窗口当前渲染高度，取不到时返回保守默认值。"""

        render_info = getattr(window, "render_info", None)
        if render_info is None:
            return 12
        return max(1, render_info.window_height)

    def _update_view_mode_from_viewport(self) -> None:
        """根据当前真实视口同步 latest/history 状态。"""

        self.state.main_view_top_display_line = self.main_window.vertical_scroll
        if self.is_main_at_bottom():
            self.state.mark_latest_view()
            return
        self.state.mark_history_view()
