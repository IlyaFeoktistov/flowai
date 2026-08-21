"""Web equivalent of ui/app.py's FlowAIApp for tools/confirm.py's `_app`
slot (connect_app()) — confirm.py already delegates permission/ask_user
prompts to whatever `_app` is connected instead of reading raw terminal
stdin, so this is the ONLY integration point needed to make bash/write
confirmations and ask_user questions work over a websocket instead of a
curses dialog."""
import asyncio
import uuid


class WebBridge:
    def __init__(self, send_json):
        self._send_json = send_json
        self._pending: dict[str, asyncio.Future] = {}

    def resolve(self, message: dict) -> bool:
        """Called from the websocket's receive loop when a
        permission_response/ask_user_response comes back from the browser."""
        future = self._pending.pop(message.get("id"), None)
        if future is None or future.done():
            return False
        future.set_result(message.get("answer"))
        return True

    def cancel_all(self) -> None:
        """The websocket disconnected mid-dialog — deny/dismiss whatever was
        still waiting instead of leaving the agent turn hung forever."""
        for future in self._pending.values():
            if not future.done():
                future.set_result(None)
        self._pending.clear()

    async def show_permission_dialog(self, action: str, detail: str) -> str:
        request_id = uuid.uuid4().hex
        future = asyncio.get_event_loop().create_future()
        self._pending[request_id] = future
        await self._send_json({
            "type": "permission_request", "id": request_id,
            "action": action, "detail": detail,
        })
        answer = await future
        return answer if answer in ("y", "a", "n") else "n"

    async def show_ask_user_dialog(self, question: str, options: list[dict], recommended: str | None) -> str | None:
        request_id = uuid.uuid4().hex
        future = asyncio.get_event_loop().create_future()
        self._pending[request_id] = future
        await self._send_json({
            "type": "ask_user_request", "id": request_id,
            "question": question, "options": options, "recommended": recommended,
        })
        return await future
