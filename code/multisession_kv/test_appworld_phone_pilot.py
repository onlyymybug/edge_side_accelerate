import unittest
from pathlib import Path

from appworld_phone_pilot import (
    call_tool, normalize_action_answer, parse_call, prepare_parameters, render,
    validate_profile_instruction,
)


class FakeWorld:
    def __init__(self):
        self.code = None

    def execute(self, code):
        self.code = code
        return "ok"


def doc(name, *properties):
    return {"name": name, "parameters": {"properties": {key: {} for key in properties}}}


class PilotAdapterTests(unittest.TestCase):
    def test_profile_rejects_wrong_sort_direction(self):
        validate_profile_instruction(
            "most_played_song", "What is the most played song by Velvet Echo?")
        with self.assertRaisesRegex(ValueError, "requires a 'most played song'"):
            validate_profile_instruction(
                "most_played_song", "What is the least played song by Zoey James?")
        validate_profile_instruction(
            "queue_popular_songs",
            "Add all the songs from Lily Moon that have been played over 980 times to my Spotify player queue.")
        with self.assertRaises(ValueError):
            validate_profile_instruction("queue_popular_songs", "Name a song by Lily Moon")

    def test_flat_parameters_repair_requires_exact_tool_schema(self):
        allowed = {"spotify__search_artists": {"query"}}
        parsed = parse_call(
            '{"name":"spotify__search_artists","query":"Velvet Echo"}',
            flat_parameter_names=allowed)
        self.assertEqual(parsed[:3],
                         ("spotify__search_artists", {"query": "Velvet Echo"}, True))
        with self.assertRaises(ValueError):
            parse_call('{"name":"spotify__search_artists","artist_id":25}',
                       flat_parameter_names=allowed)

    def test_narrow_json_repair(self):
        self.assertEqual(
            parse_call('{"name":"spotify__show_current_song","parameters:{}}<|im_end|>'),
            ("spotify__show_current_song", {}, True, 0),
        )
        self.assertEqual(
            parse_call('{"name":"spotify__previous_song","parameters {}"}<|im_end|>'),
            ("spotify__previous_song", {}, True, 0),
        )
        self.assertEqual(
            parse_call('{"name":"spotify__next_song"}', {"spotify__next_song"}),
            ("spotify__next_song", {}, True, 0),
        )
        with self.assertRaises(ValueError):
            parse_call('{"name":"spotify__show_artist","parameters":{"artist_id":1},"x":1}')

    def test_token_only_attached_to_api_that_accepts_it(self):
        visible = {
            "spotify__show_current_song": doc("spotify__show_current_song"),
            "spotify__show_artist": doc("spotify__show_artist", "artist_id"),
        }
        execution = {
            "spotify__show_current_song": doc("spotify__show_current_song", "access_token"),
            "spotify__show_artist": doc("spotify__show_artist", "artist_id"),
        }
        params, dropped = prepare_parameters(
            "spotify__show_current_song", {}, visible, execution, "secret")
        self.assertEqual(params, {"access_token": "secret"})
        self.assertEqual(dropped, [])
        params, dropped = prepare_parameters(
            "spotify__show_artist", {"artist_id": 24, "song_id": 238},
            visible, execution, "secret")
        self.assertEqual(params, {"artist_id": 24})
        self.assertEqual(dropped, ["song_id"])

    def test_qwen_hermes_tool_format(self):
        self.assertEqual(parse_call(
            '<tool_call>\n{"name":"spotify__next_song","arguments":{}}\n'
            '</tool_call><|im_end|>', {"spotify__next_song"}, "qwen_hermes"),
            ("spotify__next_song", {}, False, 0))
        self.assertEqual(parse_call(
            '<tool_call>{"name":"spotify__next_song","arguments":{}}</tool_call>'
            '<tool_call>{"name":"spotify__next_song","arguments":{}}</tool_call>',
            {"spotify__next_song"}, "qwen_hermes"),
            ("spotify__next_song", {}, False, 1))
        self.assertEqual(parse_call(
            '<tool_call>\n"name":"spotify__next_song","arguments":{}\n</tool_call>',
            {"spotify__next_song"}, "qwen_bundle_template"),
            ("spotify__next_song", {}, True, 0))
        prompt = render([{"role": "user", "content": "Next song"}],
                        [doc("spotify__next_song")], "token", "song_navigation",
                        "qwen_hermes")
        self.assertIn("<tool_call>", prompt)
        self.assertIn('"arguments": <args-json-object>', prompt)

    def test_bundle_template_renders_tool_response(self):
        template = (Path(__file__).resolve().parents[2] /
                    "qnn_multisession_phone_bundle_ctx6144/device/chat_template.jinja").read_text()
        prompt = render([
            {"role": "user", "content": "Next song"},
            {"role": "assistant", "content":
             '<tool_call>\n{"name":"spotify__next_song","arguments":{}}\n</tool_call>'},
            {"role": "tool", "content": '{"song_id":42}'},
        ], [doc("spotify__next_song")], "token", "song_navigation",
            "qwen_bundle_template", template)
        self.assertIn("<tool_response>\n{\"song_id\":42}\n</tool_response>", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))

    def test_unlisted_tool_argument_cannot_execute(self):
        world = FakeWorld()
        docs = {"spotify__show_artist": doc("spotify__show_artist", "artist_id")}
        with self.assertRaises(ValueError):
            call_tool(world, "spotify__show_artist", {"song_id": 238}, docs)
        self.assertIsNone(world.code)
        self.assertEqual(call_tool(world, "spotify__show_artist", {"artist_id": 24}, docs), "ok")
        self.assertEqual(world.code, "print(apis.spotify.show_artist(artist_id=24))")

    def test_action_completion_cannot_supply_answer(self):
        params, normalized = normalize_action_answer(
            "song_navigation", "supervisor__complete_task",
            {"answer": "I found it", "status": "success"})
        self.assertEqual(params, {"answer": None, "status": "success"})
        self.assertTrue(normalized)
        params, normalized = normalize_action_answer(
            "follower_count", "supervisor__complete_task", {"answer": "20"})
        self.assertEqual(params, {"answer": "20"})
        self.assertFalse(normalized)


if __name__ == "__main__":
    unittest.main()
