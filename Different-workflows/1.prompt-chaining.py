"""
Prompt Chaining Workflow using LangGraph with Pydantic Validation
Example: Marketing Copy Generation and Translation Pipeline

This example demonstrates how to build a prompt chaining workflow using LangGraph
with Pydantic models for structured validation to avoid parsing errors and loops.
"""

import json
import re
import os
from typing import Dict, List, Optional, TypedDict, Annotated
from dataclasses import dataclass
from enum import Enum
from IPython.display import display, Image


# Pydantic imports for structured validation
from pydantic import BaseModel, Field, ValidationError

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# Set up environment variables
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Gemini LLM import
from langchain_google_genai import ChatGoogleGenerativeAI

# Pydantic models for structured validation
class OutlineValidation(BaseModel):
    """Pydantic model for outline validation"""
    is_valid: bool
    missing_elements: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    has_hook: bool = Field(default=False)
    has_benefits: bool = Field(default=False)
    has_social_proof: bool = Field(default=False)
    has_call_to_action: bool = Field(default=False)
    proper_structure: bool = Field(default=False)

class MarketingCopyValidation(BaseModel):
    """Pydantic model for marketing copy validation"""
    is_valid: bool
    word_count: int
    has_clear_cta: bool = Field(default=False)
    has_benefits: bool = Field(default=False)
    tone_appropriate: bool = Field(default=False)
    has_subject_line: bool = Field(default=False)
    issues: List[str] = Field(default_factory=list)

class TranslationValidation(BaseModel):
    """Pydantic model for translation validation"""
    is_valid: bool
    length: int
    maintains_tone: bool = Field(default=False)
    culturally_appropriate: bool = Field(default=False)
    issues: List[str] = Field(default_factory=list)

# Initialize the actual LLM client
def create_llm_client():
    """Create and return a Gemini LLM client"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        timeout=30,
        max_retries=2
    )

# State definition for the workflow
class WorkflowState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    product_info: Dict
    outline: Optional[str]
    marketing_copy: Optional[str]
    translations: Dict[str, str]
    validation_results: Dict
    target_languages: List[str]
    current_step: str
    errors: List[str]
    retry_count: int

class MarketingCopyChain:
    """LangGraph-based prompt chaining workflow for marketing copy generation"""
    
    def __init__(self, llm_client: ChatGoogleGenerativeAI):
        self.llm = llm_client
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create the state graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes for each step
        workflow.add_node("generate_outline", self.generate_outline)
        workflow.add_node("validate_outline", self.validate_outline)
        workflow.add_node("generate_marketing_copy", self.generate_marketing_copy)
        workflow.add_node("validate_marketing_copy", self.validate_marketing_copy)
        workflow.add_node("translate_copy", self.translate_copy)
        workflow.add_node("validate_translations", self.validate_translations)
        workflow.add_node("finalize_results", self.finalize_results)
        
        # Define the workflow edges
        workflow.add_edge(START, "generate_outline")
        workflow.add_edge("generate_outline", "validate_outline")
        
        # Conditional edge after outline validation
        workflow.add_conditional_edges(
            "validate_outline",
            self.should_retry_outline,
            {
                "proceed": "generate_marketing_copy",
                "retry": "generate_outline",
                "fail": END
            }
        )
        
        workflow.add_edge("generate_marketing_copy", "validate_marketing_copy")
        
        # Conditional edge after marketing copy validation
        workflow.add_conditional_edges(
            "validate_marketing_copy",
            self.should_retry_marketing_copy,
            {
                "proceed": "translate_copy",
                "retry": "generate_marketing_copy",
                "fail": END
            }
        )
        
        workflow.add_edge("translate_copy", "validate_translations")
        workflow.add_edge("validate_translations", "finalize_results")
        workflow.add_edge("finalize_results", END)

        graph= workflow.compile(checkpointer=MemorySaver())
        # Display the graph in Mermaid format
        try:
            graph_image = graph.get_graph(xray=True).draw_mermaid_png()
            with open("workflow_graph.png", "wb") as f:
                f.write(graph_image)
            print("✅ Graph saved as 'workflow_graph.png'")
            print("📂 You can now open the PNG file to view the workflow")
        except Exception as e:
            print(f"❌ Error saving graph: {e}")
        
        return graph
    
    def _get_structured_response(self, prompt: str, response_model: BaseModel, max_retries: int = 2):
        """Get structured response using Pydantic model with retry logic."""
        for attempt in range(max_retries + 1):
            try:
                # Add structured output instructions to the prompt
                structured_prompt = f"""
                {prompt}
                
                IMPORTANT: Return ONLY a valid JSON object that matches this schema:
                {response_model.model_json_schema()}
                
                Do not include any explanation or additional text - just the JSON object.
                """
                
                messages = [HumanMessage(content=structured_prompt)]
                response = self.llm.invoke(messages)
                response_text = response.content.strip()
                
                # Clean up the response to extract JSON
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end != -1:
                    json_str = response_text[json_start:json_end]
                    parsed_data = json.loads(json_str)
                    # Validate with Pydantic
                    return response_model.model_validate(parsed_data)
                else:
                    raise ValueError("No valid JSON found in response")
                    
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                print(f"⚠️ Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries:
                    # Return a default "failed" response
                    if response_model == OutlineValidation:
                        return OutlineValidation(
                            is_valid=False,
                            missing_elements=[f"Validation failed after {max_retries + 1} attempts"],
                            suggestions=["Please regenerate the outline"]
                        )
                    elif response_model == MarketingCopyValidation:
                        return MarketingCopyValidation(
                            is_valid=False,
                            word_count=0,
                            issues=[f"Validation failed after {max_retries + 1} attempts"]
                        )
                    elif response_model == TranslationValidation:
                        return TranslationValidation(
                            is_valid=False,
                            length=0,
                            issues=[f"Validation failed after {max_retries + 1} attempts"]
                        )
                
                # Wait a bit before retrying
                import time
                time.sleep(1)
    
    def generate_outline(self, state: WorkflowState) -> WorkflowState:
        """Node: Generate marketing copy outline"""
        print("🔄 Generating outline...")
        
        product_info = state["product_info"]
        
        prompt = f"""
        Create a detailed outline for marketing copy for the following product:
        
        Product: {product_info['name']}
        Description: {product_info['description']}
        Target Audience: {product_info['target_audience']}
        Key Features: {', '.join(product_info['features'])}
        
        The outline should include:
        1. Hook/opening that addresses customer pain
        2. Product introduction with unique value prop
        3. Key benefits (not just features)
        4. Social proof elements
        5. Strong call to action with urgency
        
        Format as a numbered list with sub-points.
        """
        
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        outline = response.content
        
        print(f"Generated outline:\n{outline}\n")
        # Reset retry count when generating new content
        return {
            **state,
            "messages": state["messages"] + [HumanMessage(content=prompt), AIMessage(content=outline)],
            "outline": outline,
            "current_step": "generate_outline",
            "retry_count": 0
        }
    
    def validate_outline(self, state: WorkflowState) -> WorkflowState:
        """Node: Validate the generated outline using Pydantic"""
        print("✅ Validating outline...")
        
        outline = state["outline"]
        
        validation_prompt = f"""
        Validate this marketing copy outline:
        
        OUTLINE:
        {outline}
        
        Check for:
        1. Has clear hook/opening (has_hook)
        2. Includes product benefits not just features (has_benefits)
        3. Has social proof section (has_social_proof)
        4. Includes strong call to action (has_call_to_action)
        5. Proper structure and flow (proper_structure)
        
        Determine if the outline is valid overall (is_valid) and list any missing elements and suggestions.
        """
        
        validation_result = self._get_structured_response(validation_prompt, OutlineValidation)
        
        return {
            **state,
            "validation_results": {**state.get("validation_results", {}), "outline": validation_result.model_dump()},
            "current_step": "validate_outline"
        }
    
    def should_retry_outline(self, state: WorkflowState) -> str:
        """Conditional edge: Determine if outline should be retried"""
        validation = state["validation_results"].get("outline", {})
        retry_count = state.get("retry_count", 0)
        
        print(f"🔍 Outline validation: {validation.get('is_valid', False)}, retry count: {retry_count}")
        
        if validation.get("is_valid", False):
            return "proceed"
        elif retry_count < 2:  # Max 2 retries
            return "retry"
        else:
            print("❌ Max retries reached for outline generation")
            return "fail"
    
    def generate_marketing_copy(self, state: WorkflowState) -> WorkflowState:
        """Node: Generate marketing copy based on outline"""
        print("📝 Generating marketing copy...")
        
        outline = state["outline"]
        product_info = state["product_info"]
        
        # Increment retry count if we're retrying
        retry_count = state.get("retry_count", 0)
        if state.get("current_step") == "validate_marketing_copy":
            retry_count += 1
        
        prompt = f"""
        Based on this outline, write compelling marketing copy:
        
        OUTLINE:
        {outline}
        
        PRODUCT INFO:
        Product: {product_info['name']}
        Description: {product_info['description']}
        Target Audience: {product_info['target_audience']}
        
        Requirements:
        - Engaging, persuasive tone
        - Strong emotional hook
        - Focus on benefits, not features
        - Include clear call-to-action
        - Keep under 200 words
        - Include subject line for email
        
        {'IMPORTANT: Address the validation issues from previous attempt.' if retry_count > 0 else ''}
        """
        
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        marketing_copy = response.content
        
        return {
            **state,
            "messages": state["messages"] + [HumanMessage(content=prompt), AIMessage(content=marketing_copy)],
            "marketing_copy": marketing_copy,
            "current_step": "generate_marketing_copy",
            "retry_count": retry_count
        }
    
    def validate_marketing_copy(self, state: WorkflowState) -> WorkflowState:
        """Node: Validate marketing copy quality using Pydantic"""
        print("🔍 Validating marketing copy...")
        
        marketing_copy = state["marketing_copy"]
        
        validation_prompt = f"""
        Validate this marketing copy:
        
        MARKETING COPY:
        {marketing_copy}
        
        Check for:
        1. Word count under 200 (word_count)
        2. Has clear call-to-action (has_clear_cta)
        3. Includes benefits not just features (has_benefits)
        4. Appropriate tone for target audience (tone_appropriate)
        5. Has subject line (has_subject_line)
        
        Count the exact number of words and determine if the copy is valid overall.
        List any specific issues found.
        """
        
        validation_result = self._get_structured_response(validation_prompt, MarketingCopyValidation)
        
        # Double-check word count manually if needed
        actual_word_count = len(marketing_copy.split())
        if validation_result.word_count == 0:
            validation_result.word_count = actual_word_count
        
        return {
            **state,
            "validation_results": {**state.get("validation_results", {}), "marketing_copy": validation_result.model_dump()},
            "current_step": "validate_marketing_copy"
        }
    
    def should_retry_marketing_copy(self, state: WorkflowState) -> str:
        """Conditional edge: Determine if marketing copy should be retried"""
        validation = state["validation_results"].get("marketing_copy", {})
        retry_count = state.get("retry_count", 0)
        
        print(f"🔍 Marketing copy validation: {validation.get('is_valid', False)}, retry count: {retry_count}")
        
        if validation.get("is_valid", False):
            return "proceed"
        elif retry_count < 2:
            return "retry"
        else:
            print("❌ Max retries reached for marketing copy generation")
            return "fail"
    
    def translate_copy(self, state: WorkflowState) -> WorkflowState:
        """Node: Translate marketing copy to target languages"""
        print("🌐 Translating marketing copy...")
        
        marketing_copy = state["marketing_copy"]
        target_languages = state["target_languages"]
        translations = {}
        
        for language in target_languages:
            prompt = f"""
            Translate this marketing copy to {language}:
            
            ORIGINAL COPY:
            {marketing_copy}
            
            Requirements:
            - Maintain the same tone and persuasive power
            - Keep cultural context appropriate
            - Preserve the call-to-action strength
            - Ensure natural flow in target language
            - Keep the same emotional impact
            """
            
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            translation = response.content
            translations[language] = translation
        
        return {
            **state,
            "translations": translations,
            "current_step": "translate_copy"
        }
    
    def validate_translations(self, state: WorkflowState) -> WorkflowState:
        """Node: Validate translation quality using Pydantic"""
        print("🔍 Validating translations...")
        
        translation_validation = {}
        
        for language, translation in state["translations"].items():
            validation_prompt = f"""
            Validate this translation quality:
            
            ORIGINAL LANGUAGE: English
            TARGET LANGUAGE: {language}
            TRANSLATION: {translation}
            
            Check for:
            1. Translation appears complete and proper length (is_valid)
            2. Character/word count (length)
            3. Maintains original tone and persuasive power (maintains_tone)
            4. Culturally appropriate for target language (culturally_appropriate)
            
            List any specific issues found.
            """
            
            validation_result = self._get_structured_response(validation_prompt, TranslationValidation)
            
            # Ensure length is set
            if validation_result.length == 0:
                validation_result.length = len(translation)
            
            translation_validation[language] = validation_result.model_dump()
        
        return {
            **state,
            "validation_results": {**state.get("validation_results", {}), "translations": translation_validation},
            "current_step": "validate_translations"
        }
    
    def finalize_results(self, state: WorkflowState) -> WorkflowState:
        """Node: Finalize and format results"""
        print("✨ Finalizing results...")
        
        return {
            **state,
            "current_step": "completed"
        }
    
    def run(self, product_info: Dict, target_languages: List[str]) -> Dict:
        """Run the complete workflow"""
        
        initial_state = {
            "messages": [],
            "product_info": product_info,
            "outline": None,
            "marketing_copy": None,
            "translations": {},
            "validation_results": {},
            "target_languages": target_languages,
            "current_step": "starting",
            "errors": [],
            "retry_count": 0
        }
        
        # Execute the workflow
        config = {"configurable": {"thread_id": "marketing_chain_1"}}
        final_state = self.graph.invoke(initial_state, config)
        
        return {
            "success": final_state["current_step"] == "completed",
            "outline": final_state.get("outline"),
            "marketing_copy": final_state.get("marketing_copy"),
            "translations": final_state.get("translations", {}),
            "validation_results": final_state.get("validation_results", {}),
            "workflow_state": final_state
        }

# Example usage
def main():
    """Example usage of the marketing copy chain"""
    
    # Initialize the workflow with real Gemini LLM
    llm_client = create_llm_client()
    marketing_chain = MarketingCopyChain(llm_client)
    
    # Product information
    product_info = {
        "name": "FreshBrew Coffee",
        "description": "Premium artisan coffee blend made from sustainably sourced beans",
        "target_audience": "Coffee enthusiasts aged 25-45 who value quality and sustainability",
        "features": [
            "Organic certified beans",
            "Small batch roasted",
            "Direct trade sourcing",
            "Rich, smooth flavor profile",
            "Eco-friendly packaging"
        ]
    }
    
    # Target languages for translation
    target_languages = ["Spanish", "French"]
    
    print("🚀 Starting Marketing Copy Generation Workflow...")
    print("=" * 60)
    
    try:
        # Run the workflow
        results = marketing_chain.run(product_info, target_languages)
        
        print("\n" + "=" * 60)
        print("📊 WORKFLOW RESULTS")
        print("=" * 60)
        
        if results["success"]:
            print("✅ Workflow completed successfully!")
            
            print("\n📋 OUTLINE:")
            print(results["outline"])
            
            print("\n📝 MARKETING COPY:")
            print(results["marketing_copy"])
            
            print("\n🌐 TRANSLATIONS:")
            for language, translation in results["translations"].items():
                print(f"\n--- {language.upper()} ---")
                print(translation)
            
            print("\n🔍 VALIDATION RESULTS:")
            for step, validation in results["validation_results"].items():
                print(f"{step}: {validation}")
        
        else:
            print("❌ Workflow failed")
            print("Errors:", results.get("errors", []))
            
    except Exception as e:
        print(f"❌ Error running workflow: {str(e)}")
        print("Make sure you have set up your GOOGLE_API_KEY in your .env file")

if __name__ == "__main__":
    main()