# 🎨 Logo Generation Pipeline with LangChain Agents

A sophisticated pipeline that uses two LangChain agents to generate and evaluate logo designs for clubs and organizations. The system combines AI-powered creativity with structured evaluation to produce high-quality logo recommendations.

## 🚀 Features

### **Generator Agent**
- Creates multiple unique logo designs based on club descriptions
- Considers personal vision and style preferences
- Generates detailed design prompts for DALL-E integration
- Provides comprehensive design metadata (style notes, target audience, approach)

### **Judge Agent**
- Evaluates logos using structured criteria (Clarity, Relevance, Creativity, Simplicity)
- Provides detailed scoring on a 1-10 scale
- Offers constructive feedback with strengths and improvement areas
- Selects the best logo with comprehensive reasoning

### **Pipeline Features**
- **Structured Evaluation**: Systematic scoring with weighted criteria
- **Multiple Logo Candidates**: Generates 3-5 diverse design options
- **AI-Powered Analysis**: Uses GPT-4 for intelligent evaluation
- **Configurable Weights**: Customizable evaluation criteria importance
- **Professional Output**: Clean, structured results with detailed explanations

## 🏗️ Architecture

```
Input (Club Description + Personal Vision)
           ↓
    Generator Agent
           ↓
    Multiple Logo Designs
           ↓
    Judge Agent (Evaluation)
           ↓
    Best Logo Selection
           ↓
    Final Output + Reasoning
```

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- LangChain framework
- DALL-E access for image generation

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd agentic-logo
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your OpenAI API key:**
   ```bash
   # Edit api_config.py with your API key
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 🎯 Usage

### **Easy Launcher (Recommended)**
```bash
# Interactive launcher with examples
python run_pipeline.py
```

### **Basic Usage**

```python
from logo_pipeline import LogoGenerationPipeline

# Create pipeline
pipeline = LogoGenerationPipeline()

# Define club description
club_description = """
SCALE (Student Club for AI and Learning Excellence) is a university organization dedicated to:
- Promoting artificial intelligence education and research
- Fostering collaboration between students interested in AI and machine learning
- Organizing workshops, hackathons, and networking events
"""

# Run pipeline
result = pipeline.run_pipeline(club_description)
```

### **Using Pre-defined Examples**
```python
from examples import SCALE_CLUB, ENVIRONMENTAL_CLUB

# Use AI/Learning club example
club_desc = SCALE_CLUB['club_description']
personal_vision = SCALE_CLUB['personal_vision']

# Run pipeline
result = pipeline.run_pipeline(club_desc, personal_vision)
```

### **With Personal Vision**

```python
personal_vision = """
I envision a logo that represents:
- The interconnected nature of AI and learning
- Growth and progress in knowledge
- Modern, tech-forward aesthetic
- Professional yet approachable appearance
"""

result = pipeline.run_pipeline(club_description, personal_vision)
```



## ⚙️ Configuration

The pipeline is highly configurable through the `api_config.py` file:

```python
# Evaluation criteria weights
EVALUATION_CRITERIA_WEIGHTS = {
    "clarity": 0.25,      # 25% weight
    "relevance": 0.30,    # 30% weight
    "creativity": 0.25,   # 25% weight
    "simplicity": 0.20    # 20% weight
}

# Logo generation settings
MAX_LOGO_CANDIDATES = 5

# Model configuration
OPENAI_MODEL = "gpt-4"
DALLE_MODEL = "dall-e-3"
```

## 📊 Evaluation Criteria

### **Scoring System (1-10 Scale)**

| Score | Rating | Description |
|-------|--------|-------------|
| 1-3   | Poor   | Significant issues or misalignment |
| 4-6   | Fair   | Some good elements but room for improvement |
| 7-8   | Good   | Solid design with minor areas for enhancement |
| 9-10  | Excellent | Outstanding design that meets all criteria |

### **Evaluation Dimensions**

1. **Clarity** - How easily the logo communicates its message
2. **Relevance** - How well it aligns with the club's mission and values
3. **Creativity** - How original and innovative the design is
4. **Simplicity** - How clean and uncluttered the design is

## 🔧 Customization

### **Adding New Evaluation Criteria**

```python
# In api_config.py
EVALUATION_CRITERIA_WEIGHTS = {
    "clarity": 0.25,
    "relevance": 0.30,
    "creativity": 0.25,
    "simplicity": 0.20,
    "uniqueness": 0.15  # New criterion
}
```

### **Modifying Agent Prompts**

```python
# Custom generation prompt - modify directly in logo_pipeline.py
# Look for the system prompts in LogoGeneratorAgent and LogoJudgeAgent classes
```

## 📁 Project Structure

```
agentic-logo/
├── logo_pipeline.py          # Main pipeline implementation
├── api_config.py             # API configuration and keys
├── run_pipeline.py           # Easy launcher with examples
├── examples.py               # Pre-defined club examples
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── docs/                     # Comprehensive documentation
│   ├── README.md            # Documentation overview
│   ├── architecture.md      # System architecture
│   ├── pipeline-design.md   # LangChain integration details
│   ├── evaluation-criteria.md # Evaluation framework
│   ├── agent-specifications.md # Agent details
│   ├── project-structure.md # Project organization
│   ├── configuration.md     # Configuration guide
│   ├── troubleshooting.md   # Common issues and solutions
│   └── SUMMARY.md          # Quick reference
└── logo_projects/           # Generated outputs (timestamped)
```

## 🚨 Troubleshooting

### **Common Issues**

1. **API Key Error**
   ```bash
   ❌ Error: OPENAI_API_KEY not found in api_config.py
   ```
   **Solution**: Edit `api_config.py` and add your OpenAI API key

2. **Import Errors**
   ```bash
   ModuleNotFoundError: No module named 'langchain'
   ```
   **Solution**: Install dependencies with `pip install -r requirements.txt`

3. **Configuration Validation Failed**
   ```bash
   ❌ Configuration validation failed
   ```
   **Solution**: Check `api_config.py` for invalid settings

### **Performance Tips**

- Use GPT-4 for better evaluation quality
- Adjust agent temperatures based on your needs
- Consider reducing `MAX_LOGO_CANDIDATES` for faster processing

## 🔮 Future Enhancements

- **Multi-modal Evaluation**: Direct image analysis using vision models
- **Style Transfer**: Apply specific design styles to generated logos
- **Batch Processing**: Generate logos for multiple clubs simultaneously
- **Export Options**: Save results in various formats (PDF, JSON, etc.)
- **Web Interface**: User-friendly web application for non-technical users

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For questions or issues, please open an issue on the repository.

---

**Built with ❤️ using LangChain and OpenAI APIs**
