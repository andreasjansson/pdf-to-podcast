import requests
from dataclasses import dataclass
import json
import subprocess
from pathlib import Path
from cog import Input, Path, include

# Include models from Replicate
llm = include("anthropic/claude-3.7-sonnet")
tts = include("minimax/speech-02-hd")
pdf_extractor = include("cuuupid/markitdown")


VOICES = [
    "Wise_Woman",
    "Friendly_Person",
    "Inspirational_girl",
    "Deep_Voice_Man",
    "Calm_Woman",
    "Casual_Guy",
    "Lively_Girl",
    "Patient_Man",
    "Young_Knight",
    "Determined_Man",
    "Lovely_Girl",
    "Decent_Boy",
    "Imposing_Manner",
    "Elegant_Man",
    "Abbess",
    "Sweet_Girl_2",
    "Exuberant_Girl",
]


@dataclass(frozen=True)
class PDFMetadata:
    filename: str
    markdown: str
    type: str  # "main" or "context"


@dataclass(frozen=True)
class DialogueEntry:
    text: str
    speaker: str


@dataclass(frozen=True)
class Conversation:
    title: str
    summary: str
    lines: list[DialogueEntry]


def pdf_to_podcast(
    pdfs: list[Path] = Input(description="PDF files to convert to podcast"),
    host_name: str = Input(description="Name of the podcast host", default="Adam"),
    guest_name: str = Input(description="Name of the podcast guest", default="Bella"),
    host_voice: str = Input(description="Voice for the podcast host", default="Patient_Man", choices=VOICES),
    guest_voice: str = Input(
        description="Voice for the podcast guest", default="Wise_Woman", choices=VOICES
    ),
    duration_minutes: int = Input(
        description="Target podcast duration in minutes", default=5, ge=1, le=20
    ),
    podcast_topic: str = Input(
        description="Optional topic guidance for the podcast", default=""
    ),
    monologue: bool = Input(
        description="Generate a monologue instead of a dialogue", default=False
    ),
) -> Path:
    """Convert PDF documents to a podcast audio file and transcript"""
    # 1. Convert PDFs to text
    pdf_metadata = process_pdfs(pdfs)

    # 2. Generate podcast content using LLM
    conversation = generate_podcast_content(
        pdf_metadata=pdf_metadata,
        host_name=host_name,
        guest_name=guest_name,
        duration_minutes=duration_minutes,
        podcast_topic=podcast_topic,
        monologue=monologue,
    )

    # 3. Convert text to speech
    audio_path = generate_audio(
        conversation=conversation,
        host_voice=host_voice,
        guest_voice=guest_voice,
        monologue=monologue,
    )

    # 4. Return results
    return audio_path


def process_pdfs(pdfs: list[Path]) -> list[PDFMetadata]:
    """Convert PDFs to text using the pdf-extractor model"""
    results = []

    for i, pdf_path in enumerate(pdfs):
        print(f"Processing PDF {i + 1}/{len(pdfs)}: {pdf_path.name}")

        # Extract text from PDF using the pdf-extractor model
        markdown_url = pdf_extractor(doc=pdf_path)
        markdown_path = Path("/tmp/result.md")
        download(markdown_url, markdown_path)

        # Create metadata for this PDF
        pdf_data = PDFMetadata(
            filename=pdf_path.name,
            markdown=markdown_path.read_text(),
            type="main" if i == 0 else "context",
        )
        results.append(pdf_data)

    return results


def generate_podcast_content(
    pdf_metadata: list[PDFMetadata],
    host_name: str,
    guest_name: str,
    duration_minutes: int,
    podcast_topic: str,
    monologue: bool,
) -> Conversation:
    """Generate podcast content using the LLM"""
    # Prepare the context from all PDFs
    all_pdf_text = ""
    for pdf in pdf_metadata:
        all_pdf_text += f"\n\nDocument: {pdf.filename}\n{pdf.markdown[:5000]}\n\n"

    # Truncate if too long
    max_context_length = 24000
    if len(all_pdf_text) > max_context_length:
        all_pdf_text = (
            all_pdf_text[:max_context_length] + "\n\n[truncated due to length]\n"
        )

    # Step 1: Generate a summary of the PDFs first
    summary_prompt = f"""Summarize the main points from these documents. Focus on key facts, figures, and insights:

    {all_pdf_text}

    Provide a concise summary in 3-5 paragraphs that captures the essential information."""

    summary = llm(prompt=summary_prompt)

    print("<<< PDF summary >>>")
    print(summary)

    # Step 2: Generate a podcast outline
    outline_prompt = f"""Based on the following summary of documents:

    {summary}

    Create an outline for a {"monologue" if monologue else "podcast conversation between a host and guest"} that discusses these documents.
    The {"speaker" if monologue else "host"} is named {host_name}{"." if monologue else f" and the guest is named {guest_name}."}.
    {"The monologue" if monologue else "The conversation"} should last approximately {duration_minutes} minutes.
    {f"The podcast should focus on: {podcast_topic}" if podcast_topic else ""}

    Create a detailed outline with 5-10 main points or segments, where each segment includes a topic and key points to discuss."""

    outline = llm(prompt=outline_prompt)
    print("<<< Podcast outline >>>")
    print(outline)

    # Step 3: Generate the actual podcast content
    if monologue:
        content_prompt = f"""You will create a monologue podcast script for {host_name} based on this outline:

        {outline}

        The monologue should:
        1. Be approximately {duration_minutes} minutes in length (about {duration_minutes * 150} words)
        2. Feel conversational and engaging
         3. Reference information from the documents summarized as: {summary}
        4. Have a clear introduction, body, and conclusion
        5. Use a natural speaking style

        Format the monologue as follows:

        {{"title": "[PODCAST TITLE]", "summary": "[BRIEF SUMMARY]", "lines": [
          {{"text": "[First line of speech]", "speaker": "{host_name}"}},
          {{"text": "[Next line of speech]", "speaker": "{host_name}"}},
          ...
        ]}}

        Return ONLY the formatted JSON with no additional text or explanation."""
    else:
        content_prompt = f"""You will create a podcast dialogue script between {host_name} and {guest_name} based on this outline:

        {outline}

        The conversation should:
        1. Be approximately {duration_minutes} minutes in length (about {duration_minutes * 150} words)
        2. Feel natural and conversational
        3. Reference information from the documents summarized as: {summary}
        4. Have the host ask questions and guide the conversation
        5. Have the guest provide expertise and insights
        6. Include back-and-forth exchanges that sound realistic

        Format the dialogue as follows:

        {{"title": "[PODCAST TITLE]", "summary": "[BRIEF SUMMARY]", "lines": [
          {{"text": "[First line of speech]", "speaker": "[{host_name} or {guest_name}]"}},
          {{"text": "[Next line of speech]", "speaker": "[{host_name} or {guest_name}]"}},
          ...
        ]}}

        Return ONLY the formatted JSON with no additional text or explanation."""

    content_response = llm(prompt=content_prompt)
    print("<<< Podcast content >>>")
    print(content_response)

    # Extract and parse the JSON from the response
    json_content = extract_json(content_response)

    # Parse the JSON data into the Conversation model
    conversation_data = json.loads(json_content)
    lines = [
        DialogueEntry(text=line["text"], speaker=line["speaker"])
        for line in conversation_data["lines"]
    ]

    return Conversation(
        title=conversation_data["title"],
        summary=conversation_data["summary"],
        lines=lines,
    )


def extract_json(content: str) -> str:
    """Extract JSON from LLM response, handling various formats"""
    # Strip markdown code blocks if present
    if "```json" in content:
        return content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        return content.split("```")[1].split("```")[0].strip()
    else:
        return content


def generate_audio(
    conversation: Conversation, host_voice: str, guest_voice: str, monologue: bool
) -> Path:
    """Generate audio from the conversation text"""
    # Create a voice mapping
    voice_mapping = {}

    # Get unique speakers
    speakers = set(entry.speaker for entry in conversation.lines)

    # Set up voice mapping
    if monologue:
        # For monologue, use only the host voice
        for speaker in speakers:
            voice_mapping[speaker] = host_voice
    else:
        # For dialogue, map the first two speakers to host and guest voices
        speaker_list = list(speakers)
        if len(speaker_list) >= 1:
            voice_mapping[speaker_list[0]] = host_voice
        if len(speaker_list) >= 2:
            voice_mapping[speaker_list[1]] = guest_voice
        # Map any additional speakers to host/guest voices alternating
        for i, speaker in enumerate(speaker_list[2:], start=2):
            voice_mapping[speaker] = host_voice if i % 2 == 0 else guest_voice

    # Process each dialogue line to create audio segments
    all_audio_segments = []

    audio_tasks = []
    for entry in conversation.lines:
        text = entry.text
        speaker = entry.speaker
        voice = voice_mapping[speaker]

        audio_task = tts.start(
            text=text,
            voice_id=voice,
            sample_rate=44100,
            english_normalization=True,
            language_boost="English"
        )
        audio_tasks.append(audio_task)

    for i, audio_task in enumerate(audio_tasks):
        audio_result = audio_task.wait()

        # Save the segment
        segment_path = Path(f"/tmp/audio-{i}.mp3")
        download(audio_result, segment_path)
        all_audio_segments.append(segment_path)

    print("TTS completed, combining audio files")

    # Combine all audio segments into a single MP3 file using FFmpeg
    combined_audio_path = Path("podcast.mp3")
    combine_audio_files(all_audio_segments, combined_audio_path)

    return combined_audio_path


def combine_audio_files(audio_files: list[Path], output_path: Path) -> None:
    """Combine multiple audio files into a single file using FFmpeg"""
    # Create a temporary file with a list of input files
    file_list_path = Path("/tmp/file_list.txt")

    with open(file_list_path, "w") as f:
        for audio_file in audio_files:
            f.write(f"file '{audio_file.absolute()}'\n")

    # Use FFmpeg to concatenate the files
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(file_list_path),
        "-c",
        "copy",
        str(output_path),
    ]

    subprocess.run(ffmpeg_cmd, check=True)


def download(url: str, path: Path) -> None:
    response = requests.get(url)
    path.write_bytes(response.content)
