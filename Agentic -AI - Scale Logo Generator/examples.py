#!/usr/bin/env python3
"""
Example club descriptions and personal visions for logo generation

This file contains pre-defined examples that can be used with the Logo Generation Pipeline
to quickly test different types of clubs and design requirements.
"""

# Example 1: AI/Learning Club
SCALE_CLUB = {
    "club_description": """
    SCALE (Student Club for AI and Learning Excellence) is a university organization dedicated to:
    - Promoting artificial intelligence education and research
    - Fostering collaboration between students interested in AI and machine learning
    - Organizing workshops, hackathons, and networking events
    - Connecting students with industry professionals and research opportunities
    - Building a community of AI enthusiasts and learners
    
    The club values innovation, collaboration, continuous learning, and practical application of AI concepts.
    """,
    
    "personal_vision": """
    I envision a logo that represents:
    - The interconnected nature of AI and learning
    - Growth and progress in knowledge
    - Modern, tech-forward aesthetic
    - Professional yet approachable appearance
    - Scalability for various use cases
    """
}

# Example 2: Environmental Club
ENVIRONMENTAL_CLUB = {
    "club_description": """
    Green Future is a student-led environmental sustainability club focused on:
    - Promoting eco-friendly practices and environmental awareness
    - Organizing campus clean-up events and sustainability workshops
    - Advocating for green initiatives and renewable energy
    - Building partnerships with local environmental organizations
    - Educating the community about climate change and conservation
    
    The club emphasizes community engagement, education, and actionable environmental solutions.
    """,
    
    "personal_vision": """
    I envision a logo that represents:
    - Connection to nature and environmental consciousness
    - Growth and renewal themes
    - Community and collaboration
    - Hope for a sustainable future
    - Professional yet approachable design
    """
}

# Example 3: Creative Arts Club
CREATIVE_ARTS_CLUB = {
    "club_description": """
    ArtFlow is a vibrant creative arts and design club that:
    - Fosters artistic expression and creative collaboration
    - Organizes art exhibitions, workshops, and design competitions
    - Provides platforms for students to showcase their creative work
    - Promotes interdisciplinary collaboration between artists and designers
    - Builds connections with local art communities and galleries
    
    The club celebrates diversity, innovation, and the transformative power of art.
    """,
    
    "personal_vision": """
    I envision a logo that represents:
    - Creative expression and artistic freedom
    - Dynamic flow and movement
    - Diversity and inclusivity
    - Modern, contemporary design aesthetic
    - Versatility across different mediums
    """
}

# Example 4: Technology Club
TECH_CLUB = {
    "club_description": """
    CodeCraft is a technology and programming club that:
    - Teaches programming languages and software development
    - Organizes coding competitions and hackathons
    - Builds real-world projects and applications
    - Connects students with tech industry professionals
    - Promotes innovation and problem-solving through technology
    
    The club focuses on practical skills, teamwork, and real-world application of technology.
    """,
    
    "personal_vision": """
    I envision a logo that represents:
    - Technology and innovation
    - Problem-solving and creativity
    - Community and collaboration
    - Modern, clean design
    - Professional appearance for industry connections
    """
}

# Example 5: Sports Club
SPORTS_CLUB = {
    "club_description": """
    ActiveLife is a multi-sport and fitness club that:
    - Promotes physical fitness and healthy living
    - Organizes sports tournaments and fitness challenges
    - Builds teamwork and leadership skills
    - Creates inclusive environment for all skill levels
    - Connects students through shared athletic interests
    
    The club emphasizes sportsmanship, fitness, and building lasting friendships.
    """,
    
    "personal_vision": """
    I envision a logo that represents:
    - Energy and movement
    - Teamwork and unity
    - Health and vitality
    - Dynamic and engaging design
    - Appeal to diverse athletic interests
    """
}

# Function to get all examples
def get_all_examples():
    """Return all available club examples."""
    return {
        "SCALE (AI/Learning)": SCALE_CLUB,
        "Green Future (Environmental)": ENVIRONMENTAL_CLUB,
        "ArtFlow (Creative Arts)": CREATIVE_ARTS_CLUB,
        "CodeCraft (Technology)": TECH_CLUB,
        "ActiveLife (Sports)": SPORTS_CLUB
    }

# Function to print example details
def print_example(name: str):
    """Print the details of a specific example."""
    examples = get_all_examples()
    if name in examples:
        example = examples[name]
        print(f"\n🎯 {name}")
        print("=" * 60)
        print("Club Description:")
        print(example["club_description"].strip())
        print("\nPersonal Vision:")
        print(example["personal_vision"].strip())
        print("=" * 60)
        return example
    else:
        print(f"❌ Example '{name}' not found.")
        print("Available examples:", list(examples.keys()))
        return None

# Function to list all examples
def list_examples():
    """List all available examples."""
    examples = get_all_examples()
    print("\n📚 Available Club Examples:")
    print("=" * 40)
    for i, name in enumerate(examples.keys(), 1):
        print(f"{i}. {name}")
    print("=" * 40)

if __name__ == "__main__":
    # Demo the examples
    list_examples()
    print("\n" + "=" * 60)
    print("To use an example in your pipeline:")
    print("from examples import SCALE_CLUB")
    print("club_desc = SCALE_CLUB['club_description']")
    print("personal_vision = SCALE_CLUB['personal_vision']")
    print("=" * 60)
