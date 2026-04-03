import argparse
import asyncio
from datetime import datetime
import yaml
from pathlib import Path


from ffmpeg import Progress
from ffmpeg.asyncio import FFmpeg
from loguru import logger
from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.models.anthropic import AnthropicModel
from strands_deepgram import deepgram

from utils import load_api_keys
"""
Records a live radio/audio stream and transcribes it using a Strands Agent
backed by the Deepgram MCP tool. Station URLs are loaded from radio_stations.yaml.
Talk radio stations in that config have been verified to stream audio immediately
on connect.
"""


class RadioSpeechToTextAgent:
    """Strands agent that records a live stream and transcribes it via Deepgram."""

    def __init__(self,):
        self.load_stations()

    def load_stations(self,):
        """Load station URLs from radio_stations.yaml into self.radio_stations.

        Returns the parsed dict (keyed by station type, then stream name) so it
        can also be used standalone before the instance is fully initialised.
        """
        radio_stations = None
        config_path = Path(__file__).parent / "radio_stations.yaml"
        with open(config_path, "r",) as f:
            radio_stations  = yaml.safe_load(f)
        self.radio_stations = radio_stations
        return radio_stations

    async def record_stream_for_seconds(self,
                                        stream_name: str = None,
                                        stream_address: str = None,
                                        seconds_to_record: int = 10
                                    ) -> str:
        """Capture `seconds_to_record` seconds of audio from `stream_address` to an mp3.

        The output file is written next to this script and named
        `<stream_name>_<YYYYmmdd_HHMMSS>.mp3`. ffmpeg stream-copies the audio
        so no re-encoding happens.

        Returns the absolute path of the recorded file.
        """
        current_path = Path(__file__).parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        OUTPUT_STREAM_NAME = f'{current_path}/{stream_name}_{timestamp}.mp3'
        logger.debug(f'OUTPUT_STREAM_NAME ({OUTPUT_STREAM_NAME})')

        ffmpeg =  FFmpeg().option('y').input(f'{stream_address}', t=seconds_to_record).output(OUTPUT_STREAM_NAME, codec='copy')
        @ffmpeg.on('progress')
        async def on_progress(progress: Progress):
            await asyncio.sleep(1)

        @ffmpeg.on("completed")
        def on_completed():
            print("Completed")
        await ffmpeg.execute()

        logger.info(f'Finished recording:\t{OUTPUT_STREAM_NAME}')
        return OUTPUT_STREAM_NAME


    async def get_radio_recording(self,
                                  station_type: str = 'talk_radio',
                                  stream_name: str = 'npr',
                                  stream_address: str = None,
                                  seconds_to_record: int = 10):
        """Record a stream and return a Deepgram transcription with speaker diarization.

        Looks up the stream URL from the loaded station config using `station_type`
        and `stream_name`, records it, then hands the file off to a Strands Agent
        that calls the Deepgram MCP tool for transcription.

        Args:
            station_type: Top-level key in radio_stations.yaml (e.g. 'talk_radio', 'music').
            stream_name: Stream key within that type (e.g. 'npr', 'kexp').
            stream_address: Unused — URL is always resolved from the config.
            seconds_to_record: How many seconds of audio to capture.

        Returns:
            The agent's response containing the transcribed text and speaker labels.
        """
        api_keys = load_api_keys()
    
        # Load radio stations
        _radio_stations = RadioSpeechToTextAgent().load_stations()
        if station_type not in _radio_stations.keys():
            logger.error(f'{station_type} is not a set of stations!')
            exit(1)
        logger.debug(_radio_stations['talk_radio'])
        
        # Record a stream
        recording = await self.record_stream_for_seconds(stream_name=stream_name, stream_address=_radio_stations[f'{station_type}'][f'{stream_name}'], seconds_to_record=seconds_to_record)

        # Use deepgram to transcribe the audio
        # agent = Agent(tools=[deepgram,])
        model = AnthropicModel(model_id="claude-haiku-4-5-20251001", max_tokens=64000)
        agent = Agent(model=model, tools=[deepgram], conversation_manager=SlidingWindowConversationManager(
                    window_size=10,
                    should_truncate_results=True,
        ))
        return agent(f'transcribe this audio: {recording} with speaker diarization')


async def main():
    """CLI entry point. Parses args and runs a single record-and-transcribe cycle."""
    parser = argparse.ArgumentParser(description='Record and transcribe a radio stream.')
    parser.add_argument('--station-type', default='talk_radio', help='Station category (default: talk_radio)')
    parser.add_argument('--stream-name', default='npr', help='Stream name within the station type (default: npr)')
    parser.add_argument('--seconds', type=int, default=10, help='Seconds to record (default: 10)')
    args = parser.parse_args()

    agent = RadioSpeechToTextAgent()
    response = await agent.get_radio_recording(
        station_type=args.station_type,
        stream_name=args.stream_name,
        seconds_to_record=args.seconds,
    )
    print(f'\n\n\nAgent Response:\n\n{response}')


if __name__ == '__main__':
    asyncio.run(main())