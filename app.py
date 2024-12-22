import streamlit as st
import whisper
from transformers import pipeline
from datetime import datetime
import os

# Title and description
st.title("Audio-to-Text Summarizer with Enhanced Key Points")
st.markdown(
    """
    Upload an audio file, and this application will:
    1. Transcribe the audio content.
    2. Generate a detailed summary with key points.
    3. Save the results into individual document files in the specified folder.
    4. Allow downloading the files through the interface.
    """
)

# Predefined folder path
output_folder = r"C:\Users\DELL\OneDrive\Desktop\output"

# Ensure the folder exists
os.makedirs(output_folder, exist_ok=True)
st.write(f"Text files will be saved in: **{output_folder}**")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file:
    # Notify the user
    st.info("Processing your audio file...")

    # Extract the file name without extension for saving
    file_name = os.path.splitext(uploaded_file.name)[0]

    # Save the uploaded file locally for processing
    local_audio_path = os.path.join(output_folder, f"{file_name}.wav")
    with open(local_audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Date, Time, Agenda, and Venue Inputs
    st.subheader("Enter Meeting Details")
    meeting_date = st.date_input("Select the date of the meeting", datetime.today().date())
    meeting_time = st.time_input("Select the time of the meeting", datetime.now().time())
    meeting_agenda = st.text_input("Enter the meeting agenda", "Discuss quarterly results")
    meeting_venue = st.text_input("Enter the meeting venue", "Conference Room A")
    meeting_datetime = f"{meeting_date} {meeting_time}"

    st.write(f"Meeting Date and Time: **{meeting_datetime}**")
    st.write(f"Meeting Agenda: **{meeting_agenda}**")
    st.write(f"Meeting Venue: **{meeting_venue}**")

    # Load Whisper model for transcription
    st.info("Transcribing the audio...")
    model = whisper.load_model("base")
    transcription_result = model.transcribe(local_audio_path)
    transcript = transcription_result["text"]

    # Display the full transcript
    st.subheader("Transcription")
    st.write(transcript)

    # Summarization with Extractive Model
    st.info("Generating summary with key points...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Summarize transcription into detailed key points
    summary_result = summarizer(
        transcript, max_length=300, min_length=100, do_sample=False
    )
    summarized_text = summary_result[0]["summary_text"]

    # Custom key points extraction from the transcript
    st.subheader("Enhanced Summary in Key Points")
    st.markdown(f"- **Meeting Date and Time:** {meeting_datetime}")
    st.markdown(f"- **Meeting Agenda:** {meeting_agenda}")
    st.markdown(f"- **Meeting Venue:** {meeting_venue}")

    # Split summarized text into sentences
    key_points = summarized_text.split(". ")
    for idx, point in enumerate(key_points):
        if point.strip():
            st.markdown(f"- {point.strip()}.")

    # Define a unique file path for each audio file
    output_file_path = os.path.join(output_folder, f"{file_name}_summary.txt")

    # Save transcription and summary to the text file
    if st.button("Save to Text File"):
        with open(output_file_path, "w", encoding="utf-8") as file:
            file.write("Audio-to-Text Summarization Report\n")
            file.write(f"\nMeeting Date and Time: {meeting_datetime}\n")
            file.write(f"Meeting Agenda: {meeting_agenda}\n")
            file.write(f"Meeting Venue: {meeting_venue}\n")
            file.write("\nTranscription:\n")
            file.write(transcript)
            file.write("\n\nSummary in Key Points:\n")
            for point in key_points:
                if point.strip():
                    file.write(f"- {point.strip()}.\n")

        st.success(f"Document saved at {output_file_path}!")

        # Allow downloading the saved file
        with open(output_file_path, "rb") as file:
            st.download_button(
                label="Download Summary as Text File",
                data=file,
                file_name=f"{file_name}_summary.txt",
                mime="text/plain",
            )
