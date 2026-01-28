import os
import unittest
import logging

from gemini_webapi import GeminiClient, TemporaryChatNotSupported, set_log_level, logger
from gemini_webapi.constants import TEMPORARY_CHAT_FLAG_INDEX
from gemini_webapi.exceptions import AuthError

logging.getLogger("asyncio").setLevel(logging.ERROR)
set_log_level("DEBUG")


class TestTemporaryChat(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.geminiclient = GeminiClient(
            os.getenv("SECURE_1PSID"), os.getenv("SECURE_1PSIDTS")
        )

        try:
            await self.geminiclient.init(auto_refresh=False)
        except AuthError as e:
            self.skipTest(e)

    async def asyncTearDown(self):
        await self.geminiclient.close()

    @logger.catch(reraise=True)
    async def test_temporary_single_request(self):
        """Test that temporary requests work and return valid responses."""
        response = await self.geminiclient.generate_content(
            "Tell me a quick fun fact about today.",
            temporary=True,
        )
        self.assertTrue(response.text)
        logger.debug(response.text)

    @logger.catch(reraise=True)
    async def test_temporary_rejects_chat_multi_turn(self):
        """Test that temporary mode is rejected in chat sessions."""
        chat = self.geminiclient.start_chat()

        with self.assertRaises(TemporaryChatNotSupported):
            await chat.send_message("Hello", temporary=True)

    @logger.catch(reraise=True)
    async def test_temporary_rejects_chat_stream(self):
        """Test that temporary mode is rejected in streaming chat sessions."""
        chat = self.geminiclient.start_chat()

        with self.assertRaises(TemporaryChatNotSupported):
            async for _ in chat.send_message_stream("Hello", temporary=True):
                pass


if __name__ == "__main__":
    unittest.main()
