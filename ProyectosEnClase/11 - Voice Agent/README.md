# Project 11: Voice Agent Development

## Overview
Build intelligent voice-enabled agents using AWS services and OpenAI's real-time voice capabilities. This project demonstrates end-to-end voice AI application development.

## Project Structure
```
11 - Voice Agent/
├── aws_controller/                        # AWS service orchestration
├── aws_lambda/                            # Serverless functions
├── streamlit-openai-realtime-voice/       # Web-based voice interface
├── .env                                   # Environment variables
└── exercise.db                            # Application database
```

## Learning Objectives
- Voice AI implementation
- Real-time speech processing
- AWS Lambda serverless architecture
- Streamlit web applications
- Natural language understanding
- Text-to-speech and speech-to-text

## Getting Started

### Prerequisites
```bash
pip install streamlit openai boto3 python-dotenv
pip install speech_recognition pydub
```

### Environment Setup
Create `.env` file with:
```env
OPENAI_API_KEY=your_openai_key
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=us-east-1
```

### How to Run

**Streamlit Voice Interface:**
```bash
cd streamlit-openai-realtime-voice
streamlit run app.py
```

**AWS Lambda Deployment:**
```bash
cd aws_lambda
# Deploy using AWS SAM or Serverless framework
```

## Key Features
- **Real-Time Voice**: Low-latency speech processing
- **Natural Conversations**: Context-aware responses
- **AWS Integration**: Scalable cloud deployment
- **Web Interface**: User-friendly Streamlit app
- **Database**: Conversation history storage
- **Multi-Language**: Support for various languages

## Architecture Components

### 1. Voice Input/Output
- Speech-to-text conversion
- Real-time audio streaming
- Text-to-speech synthesis
- Audio processing

### 2. AWS Controller
- Service orchestration
- Lambda function management
- Resource coordination

### 3. Lambda Functions
- Serverless processing
- Event-driven architecture
- Auto-scaling capabilities

### 4. Streamlit Interface
- Interactive web UI
- Real-time visualization
- User conversation flow

## Use Cases
- **Customer Service**: Voice-enabled support
- **Virtual Assistants**: Personal AI helpers
- **Accessibility**: Voice interfaces for users
- **Education**: Interactive tutoring
- **Healthcare**: Patient interaction
- **Smart Home**: Voice control systems

## Technologies
- **OpenAI**: GPT models + Real-time voice
- **AWS Lambda**: Serverless compute
- **AWS Polly/Transcribe**: Voice services
- **Streamlit**: Web framework
- **SQLite**: Conversation storage

## Features Demonstrated
- Wake word detection
- Voice activity detection
- Intent recognition
- Context management
- Conversation history
- Error handling
- Latency optimization

## Deployment
- Local development server
- AWS Lambda serverless
- Streamlit Cloud hosting
- Container deployment options
