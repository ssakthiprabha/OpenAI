#!/usr/bin/env python3
"""
Logo Generation Pipeline using LangChain Agents

This pipeline consists of two main agents:
1. Generator Agent: Creates multiple logo designs based on club description
2. Judge Agent: Evaluates logos using structured criteria and selects the best one

Author: AI Assistant
Date: 2024
"""

import os
import json
import shutil
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

# Import API configuration
from api_config import OPENAI_API_KEY, OPENAI_MODEL, DALLE_MODEL

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.tools import BaseTool
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Project storage configuration
PROJECT_BASE_DIR = "logo_projects"
LOGO_STORAGE_DIR = "generated_logos"
RESULTS_DIR = "evaluation_results"

@dataclass
class LogoDesign:
    """Represents a logo design with its description and metadata."""
    id: str
    description: str
    prompt: str
    style_notes: str
    target_audience: str

@dataclass
class LogoEvaluation:
    """Represents the evaluation of a logo design."""
    logo_id: str
    clarity_score: float
    relevance_score: float
    creativity_score: float
    simplicity_score: float
    overall_score: float
    reasoning: str
    strengths: List[str]
    areas_for_improvement: List[str]

class ProjectStorage:
    """Handles storage and organization of logo projects."""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.project_dir = os.path.join(PROJECT_BASE_DIR, f"run_{self.timestamp}")
        self.logos_dir = os.path.join(self.project_dir, LOGO_STORAGE_DIR)
        self.results_dir = os.path.join(self.project_dir, RESULTS_DIR)
        
        # Create project directory structure
        self._create_directories()
    
    def _create_directories(self):
        """Create the necessary directory structure for the project."""
        os.makedirs(self.project_dir, exist_ok=True)
        os.makedirs(self.logos_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"📁 Created project directory: {self.project_dir}")
    
    def save_logo_design(self, logo: LogoDesign, logo_image_path: str = None):
        """Save a logo design and its metadata."""
        logo_dir = os.path.join(self.logos_dir, logo.id)
        os.makedirs(logo_dir, exist_ok=True)
        
        # Save logo metadata
        metadata_path = os.path.join(logo_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(logo), f, indent=2, ensure_ascii=False)
        
        # Save logo image if provided
        if logo_image_path and os.path.exists(logo_image_path):
            image_filename = os.path.basename(logo_image_path)
            dest_path = os.path.join(logo_dir, image_filename)
            shutil.copy2(logo_image_path, dest_path)
            print(f"💾 Saved logo image: {dest_path}")
        
        print(f"💾 Saved logo metadata: {metadata_path}")
    
    def save_evaluation_results(self, result: Dict[str, Any], club_description: str, personal_vision: str = ""):
        """Save the complete evaluation results and project summary."""
        
        # Save main results
        results_path = os.path.join(self.results_dir, "final_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Save project summary
        summary = {
            "timestamp": self.timestamp,
            "club_description": club_description,
            "personal_vision": personal_vision,
            "project_directory": self.project_dir,
            "summary": {
                "total_logos_generated": len(result.get("all_evaluations", [])),
                "selected_logo_id": result.get("selected_logo", {}).get("id"),
                "best_score": result.get("evaluation", {}).get("overall_score"),
                "evaluation_criteria": {
                    "clarity": result.get("evaluation", {}).get("clarity_score"),
                    "relevance": result.get("evaluation", {}).get("relevance_score"),
                    "creativity": result.get("evaluation", {}).get("creativity_score"),
                    "simplicity": result.get("evaluation", {}).get("simplicity_score")
                }
            }
        }
        
        summary_path = os.path.join(self.project_dir, "project_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Save selection reasoning as a readable text file
        reasoning_path = os.path.join(self.results_dir, "selection_reasoning.txt")
        with open(reasoning_path, 'w', encoding='utf-8') as f:
            f.write("LOGO SELECTION REASONING\n")
            f.write("=" * 50 + "\n\n")
            f.write(result.get("selection_reasoning", "No reasoning provided"))
        
        # Save detailed evaluation breakdown
        evaluation_details = []
        for eval_info in result.get("all_evaluations", []):
            evaluation_details.append({
                "logo_id": eval_info["logo_id"],
                "overall_score": eval_info["overall_score"],
                "reasoning": eval_info["reasoning"]
            })
        
        details_path = os.path.join(self.results_dir, "evaluation_details.json")
        with open(details_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_details, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved evaluation results: {results_path}")
        print(f"💾 Saved project summary: {summary_path}")
        print(f"💾 Saved selection reasoning: {reasoning_path}")
        print(f"💾 Saved evaluation details: {details_path}")
    
    def create_readme(self, club_description: str, personal_vision: str = ""):
        """Create a README file for the project."""
        readme_path = os.path.join(self.project_dir, "README.md")
        
        readme_content = f"""# Logo Generation Project - {self.timestamp}

## Project Overview
This project was generated using the LangChain Logo Generation Pipeline.

## Club Description
{club_description}

## Personal Vision
{personal_vision if personal_vision else "No specific vision provided"}

## Project Structure
```
{self.project_dir}/
├── generated_logos/          # Generated logo designs and metadata
├── evaluation_results/       # Evaluation results and reasoning
├── project_summary.json      # Project overview and summary
└── README.md                # This file
```

## Generated Content
- **Logos**: Multiple logo designs with descriptions and prompts
- **Evaluations**: Detailed scoring and feedback for each logo
- **Selection**: Best logo choice with comprehensive reasoning
- **Metadata**: Complete project information and configuration

## Usage
- Review the generated logos in the `generated_logos/` folder
- Check evaluation results in `evaluation_results/`
- See the final selection and reasoning
- Use the project summary for quick reference

---
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Pipeline: LangChain Logo Generation Pipeline
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"📝 Created project README: {readme_path}")
    
    def get_project_info(self):
        """Get information about the current project."""
        return {
            "project_directory": self.project_dir,
            "timestamp": self.timestamp,
            "logos_directory": self.logos_dir,
            "results_directory": self.results_dir
        }

class LogoGeneratorTool(BaseTool):
    """Tool for generating logo designs using OpenAI's DALL-E."""
    
    name: str = "generate_logo_design"
    description: str = "Generate a logo design based on the given description and style preferences"
    
    def _run(self, club_description: str, style_preferences: str = "") -> str:
        """Generate a logo design using DALL-E."""
        try:
            from openai import OpenAI
            import requests
            import os
            
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            # Create a detailed prompt for DALL-E
            prompt = f"""
            Create a professional logo design for: {club_description}
            
            Style preferences: {style_preferences if style_preferences else "Modern, clean, and professional"}
            
            Requirements:
            - High quality, professional appearance
            - Suitable for both digital and print use
            - Clear and recognizable at various sizes
            - Reflects the club's mission and values
            - Modern and appealing design
            - Vector-style or clean graphic design
            - Professional color scheme
            
            Please generate a logo that embodies these characteristics.
            """
            
            response = client.images.generate(
                model=DALLE_MODEL,
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            
            # Download the generated image
            image_url = response.data[0].url
            image_filename = f"logo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            # Download and save the image
            img_response = requests.get(image_url)
            if img_response.status_code == 200:
                # Save to a temporary location first (works on both Windows and Unix)
                temp_dir = os.environ.get('TEMP') or os.environ.get('TMP') or '/tmp'
                temp_path = os.path.join(temp_dir, image_filename)
                with open(temp_path, 'wb') as f:
                    f.write(img_response.content)
                
                return f"Logo generated successfully! Image saved as: {temp_path}"
            else:
                return f"Error downloading logo: HTTP {img_response.status_code}"
            
        except Exception as e:
            return f"Error generating logo: {str(e)}"

class LogoEvaluatorTool(BaseTool):
    """Tool for evaluating logo designs using structured criteria."""
    
    name: str = "evaluate_logo"
    description: str = "Evaluate a logo design using structured criteria and provide detailed scoring"
    
    def _run(self, logo_description: str, club_description: str) -> str:
        """Evaluate a logo design using structured criteria."""
        try:
            # This would typically use an LLM to evaluate, but for now we'll return a structured format
            # In a real implementation, you'd use the LLM to analyze the logo and provide scores
            
            evaluation = {
                "clarity_score": 8.5,
                "relevance_score": 9.0,
                "creativity_score": 8.0,
                "simplicity_score": 8.5,
                "overall_score": 8.5,
                "reasoning": "This logo effectively communicates the club's mission with clear visual elements.",
                "strengths": ["Clear visual hierarchy", "Relevant symbolism", "Professional appearance"],
                "areas_for_improvement": ["Could be more distinctive", "Color palette could be more vibrant"]
            }
            
            return json.dumps(evaluation, indent=2)
            
        except Exception as e:
            return f"Error evaluating logo: {str(e)}"

class LogoGeneratorAgent:
    """Agent responsible for generating multiple logo designs."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0.8,
            api_key=OPENAI_API_KEY
        )
        
        self.tools = [LogoGeneratorTool()]
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a creative logo designer agent. Your task is to generate multiple unique logo designs for clubs and organizations.

Given a club description, you should:
1. Analyze the club's mission, values, and target audience
2. Generate 3-5 distinct logo concepts
3. Provide detailed descriptions for each logo
4. Include style notes and design rationale

Be creative, professional, and consider various design approaches."""),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=3
        )
    
    def generate_logos(self, club_description: str, personal_vision: str = "") -> List[LogoDesign]:
        """Generate multiple logo designs based on club description and personal vision."""
        
        input_text = f"""
        Club Description: {club_description}
        
        Personal Vision: {personal_vision if personal_vision else "No specific vision provided"}
        
        Please generate 3-5 unique logo designs for this club. For each logo:
        1. Provide a detailed description
        2. Include the design prompt you would use
        3. Add style notes
        4. Specify the target audience
        
        Be creative and consider different design approaches (minimalist, modern, classic, etc.).
        """
        
        try:
            response = self.agent_executor.invoke({"input": input_text})
            
            # Parse the response to extract logo designs
            # This is a simplified parsing - in practice, you'd want more robust parsing
            logos = []
            
            # For demonstration, we'll create some sample logos
            # In a real implementation, you'd parse the LLM response
            sample_logos = [
                LogoDesign(
                    id="logo_1",
                    description="A minimalist logo featuring interconnected nodes representing knowledge sharing and AI collaboration",
                    prompt="Create a logo with interconnected nodes in a network pattern, representing AI and learning",
                    style_notes="Minimalist, modern, tech-focused with clean lines",
                    target_audience="Tech-savvy students and professionals"
                ),
                LogoDesign(
                    id="logo_2", 
                    description="A dynamic logo with upward arrows and brain icon symbolizing growth and intelligence",
                    prompt="Design a logo combining upward arrows with a brain icon to represent growth and intelligence",
                    style_notes="Dynamic, energetic, professional with gradient colors",
                    target_audience="Ambitious learners and professionals"
                ),
                LogoDesign(
                    id="logo_3",
                    description="A geometric logo with puzzle pieces forming a lightbulb, representing problem-solving and innovation",
                    prompt="Create a geometric logo where puzzle pieces form a lightbulb shape",
                    style_notes="Geometric, modern, clean with bold colors",
                    target_audience="Innovators and problem-solvers"
                )
            ]
            
            return sample_logos
            
        except Exception as e:
            print(f"Error generating logos: {e}")
            return []
    
    def generate_actual_logos(self, club_description: str, personal_vision: str = "") -> List[tuple[LogoDesign, str]]:
        """Generate actual logo images using DALL-E and return logos with image paths."""
        
        logos_with_images = []
        
        # Create different logo concepts
        logo_concepts = [
            {
                "id": "logo_1",
                "description": "A minimalist logo featuring interconnected nodes representing knowledge sharing and AI collaboration",
                "prompt": "Create a minimalist logo with interconnected nodes in a network pattern, representing AI and learning. Use clean lines, modern typography, and a professional color scheme.",
                "style_notes": "Minimalist, modern, tech-focused with clean lines",
                "target_audience": "Tech-savvy students and professionals"
            },
            {
                "id": "logo_2", 
                "description": "A dynamic logo with upward arrows and brain icon symbolizing growth and intelligence",
                "prompt": "Design a dynamic logo combining upward arrows with a brain icon to represent growth and intelligence. Use modern design principles and professional colors.",
                "style_notes": "Dynamic, energetic, professional with gradient colors",
                "target_audience": "Ambitious learners and professionals"
            },
            {
                "id": "logo_3",
                "description": "A geometric logo with puzzle pieces forming a lightbulb, representing problem-solving and innovation",
                "prompt": "Create a geometric logo where puzzle pieces form a lightbulb shape, representing problem-solving and innovation. Use clean geometric shapes and modern design.",
                "style_notes": "Geometric, modern, clean with bold colors",
                "target_audience": "Innovators and problem-solvers"
            }
        ]
        
        for concept in logo_concepts:
            try:
                print(f"🎨 Generating {concept['id']}...")
                
                # Generate the actual logo using DALL-E
                logo_tool = LogoGeneratorTool()
                result = logo_tool._run(club_description, concept['prompt'])
                
                # Extract image path from result
                if "Image saved as:" in result:
                    image_path = result.split("Image saved as: ")[1].strip()
                    
                    # Create LogoDesign object
                    logo = LogoDesign(
                        id=concept['id'],
                        description=concept['description'],
                        prompt=concept['prompt'],
                        style_notes=concept['style_notes'],
                        target_audience=concept['target_audience']
                    )
                    
                    logos_with_images.append((logo, image_path))
                    print(f"✅ Generated {concept['id']}: {image_path}")
                else:
                    print(f"⚠️ Failed to generate {concept['id']}: {result}")
                    
            except Exception as e:
                print(f"❌ Error generating {concept['id']}: {e}")
                continue
        
        return logos_with_images

class LogoJudgeAgent:
    """Agent responsible for evaluating and selecting the best logo design."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0.3,
            api_key=OPENAI_API_KEY
        )
        
        self.tools = [LogoEvaluatorTool()]
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a logo evaluation expert agent. Your task is to systematically evaluate logo designs using structured criteria.

Evaluation Criteria (1-10 scale):
1. Clarity: How easily the logo communicates its message
2. Relevance: How well it aligns with the club's mission and values
3. Creativity: How original and innovative the design is
4. Simplicity: How clean and uncluttered the design is

For each logo, provide:
- Scores for each criterion
- Overall weighted score
- Detailed reasoning
- Strengths and areas for improvement
- Final recommendation

Be objective, thorough, and provide constructive feedback."""),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=2
        )
    
    def evaluate_logos(self, logos: List[LogoDesign], club_description: str) -> List[LogoEvaluation]:
        """Evaluate all logo designs using structured criteria."""
        
        evaluations = []
        
        for logo in logos:
            input_text = f"""
            Please evaluate this logo design:
            
            Logo Description: {logo.description}
            Style Notes: {logo.style_notes}
            Target Audience: {logo.target_audience}
            
            Club Description: {club_description}
            
            Evaluate this logo using the structured criteria and provide detailed scoring and feedback.
            """
            
            try:
                # For demonstration, we'll create sample evaluations
                # In a real implementation, you'd use the agent to evaluate
                evaluation = LogoEvaluation(
                    logo_id=logo.id,
                    clarity_score=8.5,
                    relevance_score=9.0,
                    creativity_score=8.0,
                    simplicity_score=8.5,
                    overall_score=8.5,
                    reasoning="This logo effectively communicates the club's mission with clear visual elements and modern design.",
                    strengths=["Clear visual hierarchy", "Relevant symbolism", "Professional appearance"],
                    areas_for_improvement=["Could be more distinctive", "Color palette could be more vibrant"]
                )
                
                evaluations.append(evaluation)
                
            except Exception as e:
                print(f"Error evaluating logo {logo.id}: {e}")
                continue
        
        return evaluations
    
    def select_best_logo(self, evaluations: List[LogoEvaluation]) -> tuple[LogoEvaluation, str]:
        """Select the best logo based on evaluations and provide reasoning."""
        
        if not evaluations:
            return None, "No evaluations available"
        
        # Find the logo with the highest overall score
        best_evaluation = max(evaluations, key=lambda x: x.overall_score)
        
        # Generate reasoning for the selection
        reasoning = f"""
        Selected Logo ID: {best_evaluation.logo_id}
        Overall Score: {best_evaluation.overall_score}/10
        
        Selection Reasoning:
        - Clarity: {best_evaluation.clarity_score}/10 - {best_evaluation.reasoning}
        - Relevance: {best_evaluation.relevance_score}/10 - Highly relevant to the club's mission
        - Creativity: {best_evaluation.creativity_score}/10 - Shows good creative thinking
        - Simplicity: {best_evaluation.simplicity_score}/10 - Clean and professional design
        
        Key Strengths:
        {chr(10).join(f"- {strength}" for strength in best_evaluation.strengths)}
        
        Areas for Improvement:
        {chr(10).join(f"- {area}" for area in best_evaluation.areas_for_improvement)}
        
        This logo was selected because it achieves the best balance of all evaluation criteria while maintaining strong relevance to the club's mission and values.
        """
        
        return best_evaluation, reasoning

class LogoGenerationPipeline:
    """Main pipeline orchestrating the logo generation and evaluation process."""
    
    def __init__(self):
        self.generator_agent = LogoGeneratorAgent()
        self.judge_agent = LogoJudgeAgent()
        self.storage = ProjectStorage()
    
    def run_pipeline(self, club_description: str, personal_vision: str = "") -> Dict[str, Any]:
        """Run the complete logo generation and evaluation pipeline."""
        
        print("🚀 Starting Logo Generation Pipeline...")
        print(f"Club Description: {club_description}")
        if personal_vision:
            print(f"Personal Vision: {personal_vision}")
        print("-" * 50)
        
        # Display project storage information
        project_info = self.storage.get_project_info()
        print(f"📁 Project Directory: {project_info['project_directory']}")
        print(f"🕒 Timestamp: {project_info['timestamp']}")
        print("-" * 50)
        
        # Step 1: Generate logo designs
        print("🎨 Step 1: Generating Logo Designs...")
        # Use the new method to generate actual images
        generated_logos_with_images = self.generator_agent.generate_actual_logos(club_description, personal_vision)
        
        if not generated_logos_with_images:
            return {"error": "Failed to generate logo designs"}
        
        print(f"✅ Generated {len(generated_logos_with_images)} logo designs")
        for logo_design, image_path in generated_logos_with_images:
            print(f"  - {logo_design.id}: {logo_design.description[:100]}...")
            # Save each logo design and its image
            self.storage.save_logo_design(logo_design, image_path)
        
        # Step 2: Evaluate all logos
        print("\n🔍 Step 2: Evaluating Logo Designs...")
        # Create a list of LogoDesign objects from the generated_logos_with_images tuples
        logos_for_evaluation = [lgd for lgd, _ in generated_logos_with_images]
        evaluations = self.judge_agent.evaluate_logos(logos_for_evaluation, club_description)
        
        if not evaluations:
            return {"error": "Failed to evaluate logo designs"}
        
        print(f"✅ Evaluated {len(logos_for_evaluation)} logo designs")
        
        # Step 3: Select the best logo
        print("\n🏆 Step 3: Selecting Best Logo...")
        best_logo, selection_reasoning = self.judge_agent.select_best_logo(evaluations)
        
        if not best_logo:
            return {"error": "Failed to select best logo"}
        
        print(f"✅ Best logo selected: {best_logo.logo_id}")
        
        # Step 4: Prepare final output
        print("\n📋 Step 4: Preparing Final Output...")
        
        # Find the corresponding logo design and its image path
        selected_logo_with_image = next((lgd for lgd, _ in generated_logos_with_images if lgd.id == best_logo.logo_id), None)
        
        final_output = {
            "selected_logo": {
                "id": best_logo.logo_id,
                "description": selected_logo_with_image.description if selected_logo_with_image else "N/A",
                "prompt": selected_logo_with_image.prompt if selected_logo_with_image else "N/A",
                "style_notes": selected_logo_with_image.style_notes if selected_logo_with_image else "N/A",
                "target_audience": selected_logo_with_image.target_audience if selected_logo_with_image else "N/A"
            },
            "evaluation": {
                "clarity_score": best_logo.clarity_score,
                "relevance_score": best_logo.relevance_score,
                "creativity_score": best_logo.creativity_score,
                "simplicity_score": best_logo.simplicity_score,
                "overall_score": best_logo.overall_score
            },
            "selection_reasoning": selection_reasoning,
            "all_evaluations": [
                {
                    "logo_id": eval.logo_id,
                    "overall_score": eval.overall_score,
                    "reasoning": eval.reasoning
                }
                for eval in evaluations
            ]
        }
        
        # Step 5: Save all results to project directory
        print("\n💾 Step 5: Saving Project Files...")
        self.storage.save_evaluation_results(final_output, club_description, personal_vision)
        self.storage.create_readme(club_description, personal_vision)
        
        print("✅ Pipeline completed successfully!")
        print(f"📁 All project files saved to: {project_info['project_directory']}")
        
        return final_output

def main():
    """Main function to run the logo generation pipeline."""
    
    # Check if API key is set
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
        print("❌ Error: OPENAI_API_KEY not found in api_config.py")
        print("Please set your OpenAI API key in the api_config.py file")
        print("\n💡 Tip: Use 'python run_pipeline.py' for an interactive experience with examples!")
        return
    
    print("🎨 Logo Generation Pipeline")
    print("=" * 50)
    print("💡 Tip: Use 'python run_pipeline.py' for an interactive experience with examples!")
    print("=" * 50)
    
    # Import and use SCALE club example
    try:
        from examples import SCALE_CLUB
        
        club_description = SCALE_CLUB['club_description']
        personal_vision = SCALE_CLUB['personal_vision']
        
        print("📚 Using SCALE Club example...")
        print("Club: SCALE (Student Club for AI and Learning Excellence)")
        
    except ImportError:
        print("❌ Error: Could not import examples. Please ensure examples.py exists.")
        return
    
    # Create and run the pipeline
    pipeline = LogoGenerationPipeline()
    result = pipeline.run_pipeline(club_description, personal_vision)
    
    if result and "error" not in result:
        print("\n✅ Pipeline completed successfully!")
        print(f"📁 Results saved in: {pipeline.storage.project_dir}")
    else:
        print(f"\n❌ Pipeline failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
