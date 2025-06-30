# DuoSynco Video Creation Pipeline Flow

## Overview

This document describes the complete end-to-end flow for automated video content creation, from message intake to final distribution.

## Pipeline Architecture

```
User Input → Messaging Platform → n8n → AI Processing → Content Generation → TTS → Lipsync → Distribution
```

## Detailed Flow

### 1. Message Intake Phase

**Entry Points:**

- Telegram Bot
- WhatsApp Business API
- Email (IMAP/POP3)
- Direct API webhook

**Message Types:**

- URL/Link: YouTube video, Tweet, Article
- Text Prompt: Topic request, debate subject
- Command: Specific format request (debate/news/translate)

**Flow:**

```
User Message → Platform Receiver → n8n Webhook → Message Queue
```

### 2. n8n Processing Phase

**Workflow Steps:**

1. **Message Receipt**: Webhook receives incoming message
2. **Authentication**: Verify sender/source
3. **Content Extraction**: Parse message type and content
4. **AI Prompt Cleaning**: Send to AI for standardization
5. **Data Fetching**:
   - YouTube subtitles via API
   - Tweet content via API
   - Article scraping
6. **Format Decision**: Determine content type (debate/news/translate)

**Output:**

```json
{
  "request_id": "uuid",
  "source": "telegram/whatsapp/email",
  "content_type": "debate/news/translate",
  "raw_content": "extracted text/subtitles",
  "metadata": {
    "language": "en",
    "duration": 300,
    "source_url": "https://..."
  }
}
```

### 3. Content Generation Phase

#### 3.1 Debate Format

- **Input**: Topic or article
- **Process**:
  1. AI generates two perspectives (pros/cons)
  2. Creates dialogue script with neutral moderator
  3. Assigns speakers: Man (pros), Woman (cons), or balanced
- **Output**: Timestamped dialogue script

#### 3.2 News Format

- **Input**: Article, tweet, or topic
- **Process**:
  1. AI summarizes into news format
  2. Creates anchor script
  3. User choice: Male or female presenter
- **Output**: News presentation script

#### 3.3 Translation Format

- **Input**: YouTube video with subtitles
- **Process**:
  1. Extract original subtitles
  2. Translate maintaining timing
  3. Random assignment: Male or female voice
- **Output**: Translated subtitle script

### 4. Audio Production Phase

**Using ElevenLabs TTS:**

1. **Voice Selection**:

   - Debate: 2 distinct voices
   - News: 1 professional voice
   - Translation: 1 voice matching content tone

2. **Audio Generation**:

   ```
   Script → ElevenLabs API → High-quality audio (MP3/WAV)
   ```

3. **Post-processing**:
   - Normalize audio levels
   - Add gaps between speakers
   - Generate master audio file

### 5. Podcast Distribution

**For audio-only platforms:**

```
Audio File → Metadata Addition → RSS Feed → Podcast Platforms
```

**Metadata includes:**

- Title, description, episode number
- Speaker credits
- Timestamps/chapters
- Cover art

### 6. Video Production Phase

**Lipsync Provider Integration:**

#### 6.1 Template Selection

- **Horizontal (16:9)**:
  - Single person: Centered presenter
  - Two people: Side-by-side (Man left, Woman right)
- **Vertical (9:16)**:
  - Single person: Full frame presenter
  - Two people: Stacked (Woman top, Man bottom)

#### 6.2 Lipsync Processing

```
Audio + Template → Collossyan/HeyGen API → Lipsynced Video
```

#### 6.3 Subtitle Integration

- Generate from script
- Properly timed
- Option to burn into video or separate file

### 7. Distribution Phase

**Platform-specific formatting:**

#### YouTube

- Video upload via API
- Metadata: Title, description, tags
- Thumbnail generation
- Subtitle file upload
- Scheduled publishing

#### TikTok/Shorts/Reels

- Vertical format optimization
- Length constraints
- Platform-specific features

#### Other Platforms

- LinkedIn video
- Twitter/X video
- Instagram TV

## Configuration Templates

### Debate Template Configuration

```yaml
debate:
  speakers:
    moderator: "neutral_voice_id"
    pro: "male_voice_id"
    con: "female_voice_id"
  video:
    layout: "side_by_side"
    template_id: "debate_template_001"
```

### News Template Configuration

```yaml
news:
  presenter:
    male: "news_anchor_male_id"
    female: "news_anchor_female_id"
  video:
    layout: "single_presenter"
    template_id: "news_template_001"
```

### Translation Template Configuration

```yaml
translation:
  voice_pool:
    male: ["voice1", "voice2", "voice3"]
    female: ["voice4", "voice5", "voice6"]
  preserve_timing: true
```

## Automation Triggers

### Manual Trigger

- User sends message with content request
- Immediate processing

### Scheduled Trigger

- Daily news summary at set time
- Weekly debate on trending topic
- Automated translation of popular videos

### Event-based Trigger

- New video from subscribed channel
- Trending topic threshold reached
- Breaking news keywords detected

## Error Handling

1. **Message Intake Errors**: Retry with exponential backoff
2. **AI Generation Errors**: Fallback to simpler prompts
3. **TTS Errors**: Use backup voice provider
4. **Lipsync Errors**: Deliver audio-only version
5. **Distribution Errors**: Queue for manual review

## Cost Optimization

- **Batch Processing**: Group similar requests
- **Caching**: Store generated content for reuse
- **Quality Tiers**: Adjust based on content importance
- **Provider Selection**: Use cost-effective providers for each step

## Monitoring & Analytics

Track:

- Request volume by source
- Content type distribution
- Processing time per phase
- Cost per video
- Distribution reach
- Error rates

## Future Enhancements

1. **Multi-language debates**: Speakers in different languages
2. **Live content**: Real-time news generation
3. **Interactive content**: Viewer-driven topics
4. **AI avatars**: Custom presenter creation
5. **Brand integration**: Sponsored content workflow
