# 🎯 Logo Generation Pipeline - Project Overview

## 🚀 Quick Start (3 Steps)

1. **Install**: `pip install -r requirements.txt`
2. **Configure**: Edit `api_config.py` with your OpenAI API key
3. **Run**: `python run_pipeline.py`

## 📁 Optimized Project Structure

```
agentic-logo/
├── 🎨 logo_pipeline.py      # Core pipeline (Generator + Judge agents)
├── ⚙️  api_config.py        # API keys & configuration
├── 🚀 run_pipeline.py       # User-friendly launcher
├── 📚 examples.py           # 5 pre-built club examples
├── 📖 README.md             # Comprehensive guide
├── 📚 docs/                 # Technical documentation
└── 📁 logo_projects/        # Generated outputs (timestamped)
```

## 🎯 What This Project Does

**Input**: Club description + Personal vision  
**Process**: 
1. **Generator Agent** → Creates 3+ logo designs using DALL-E
2. **Judge Agent** → Evaluates logos using AI scoring (Clarity, Relevance, Creativity, Simplicity)
3. **Output**: Best logo + detailed reasoning + organized project folder

## 🔑 Key Features

- ✅ **Two LangChain Agents** working together
- ✅ **DALL-E Integration** for actual logo generation
- ✅ **Structured Evaluation** with weighted criteria
- ✅ **Organized Storage** with timestamped folders
- ✅ **5 Pre-built Examples** for quick testing
- ✅ **Easy Launcher** with interactive menu

## 📊 Evaluation Criteria (Weights)

- **Clarity**: 25% - How easily the logo communicates
- **Relevance**: 30% - Alignment with club's mission
- **Creativity**: 25% - Originality and innovation
- **Simplicity**: 20% - Clean, uncluttered design

## 🎨 Available Examples

1. **SCALE** - AI/Learning Excellence Club
2. **Green Future** - Environmental Sustainability
3. **ArtFlow** - Creative Arts & Design
4. **CodeCraft** - Technology & Programming
5. **ActiveLife** - Sports & Fitness

## 🚀 Usage Options

### Option 1: Interactive Launcher (Recommended)
```bash
python run_pipeline.py
# Choose from examples or enter custom input
```

### Option 2: Direct Python Usage
```python
from logo_pipeline import LogoGenerationPipeline
from examples import SCALE_CLUB

pipeline = LogoGenerationPipeline()
result = pipeline.run_pipeline(
    SCALE_CLUB['club_description'],
    SCALE_CLUB['personal_vision']
)
```

### Option 3: Custom Input
```python
pipeline = LogoGenerationPipeline()
result = pipeline.run_pipeline(
    "Your club description here...",
    "Your personal vision here..."
)
```

## 📁 Output Structure

Each run creates a timestamped folder:
```
logo_projects/run_20241217_143022/
├── generated_logos/
│   ├── logo_1/
│   │   ├── metadata.json
│   │   └── logo_1.png
│   ├── logo_2/
│   └── logo_3/
├── evaluation_results/
│   ├── final_results.json
│   └── project_summary.txt
└── README.md
```

## 🔧 Customization

- **Evaluation Weights**: Edit `api_config.py`
- **Agent Prompts**: Modify system prompts in `logo_pipeline.py`
- **Logo Count**: Change `MAX_LOGO_CANDIDATES` in `api_config.py`

## 📚 Documentation

- **`README.md`** - Complete project guide
- **`docs/`** - Technical documentation
  - Architecture & design patterns
  - Agent specifications
  - Pipeline integration details
  - Troubleshooting guide

## ⚡ Performance Tips

- Use GPT-4 for better evaluation quality
- Reduce `MAX_LOGO_CANDIDATES` for faster processing
- Ensure stable internet for DALL-E image generation

---

**Ready to generate amazing logos? Run `python run_pipeline.py` and get started!** 🎨✨
