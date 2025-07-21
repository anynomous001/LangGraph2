"""
Human-in-the-Loop Document Review and Approval Workflow using LangGraph
Real-world Example: Legal Contract Review and Compliance Check Pipeline

This workflow demonstrates a realistic scenario where AI assists in document review,
but human experts make critical decisions at key checkpoints before proceeding.
"""

import json
import os
from typing import Dict, List, Optional, TypedDict, Annotated, Literal
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime

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
class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class HumanDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_REVISION = "request_revision"
    ESCALATE = "escalate"

class DocumentAnalysis(BaseModel):
    """AI analysis of the document"""
    document_type: str
    key_terms_identified: List[str] = Field(default_factory=list)
    potential_issues: List[str] = Field(default_factory=list)
    compliance_concerns: List[str] = Field(default_factory=list)
    risk_level: RiskLevel
    confidence_score: float = Field(ge=0.0, le=1.0)
    summary: str
    requires_human_review: bool = Field(default=True)

class ComplianceCheck(BaseModel):
    """Compliance verification results"""
    regulations_checked: List[str] = Field(default_factory=list)
    compliance_violations: List[str] = Field(default_factory=list)
    is_compliant: bool
    risk_score: float = Field(ge=0.0, le=10.0)
    recommendations: List[str] = Field(default_factory=list)
    requires_legal_review: bool = Field(default=False)

class HumanReviewInput(BaseModel):
    """Human reviewer input structure"""
    reviewer_name: str
    reviewer_role: str
    decision: HumanDecision
    comments: str
    additional_requirements: List[str] = Field(default_factory=list)
    escalation_reason: Optional[str] = None
    timestamp: str

class RevisionRequest(BaseModel):
    """Structure for revision requests"""
    requested_changes: List[str]
    priority_level: RiskLevel
    deadline: Optional[str] = None
    specific_instructions: str

# State definition for the workflow
class DocumentWorkflowState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    document_content: str
    document_metadata: Dict
    ai_analysis: Optional[Dict]
    compliance_results: Optional[Dict]
    human_reviews: List[Dict]
    revision_requests: List[Dict]
    current_step: str
    workflow_status: str
    requires_human_input: bool
    human_input_received: bool
    retry_count: int
    final_decision: Optional[str]
    audit_trail: List[Dict]

class HumanInLoopDocumentWorkflow:
    """LangGraph-based document review workflow with human oversight"""
    
    def __init__(self, llm_client: ChatGoogleGenerativeAI):
        self.llm = llm_client
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with human-in-the-loop checkpoints"""
        
        workflow = StateGraph(DocumentWorkflowState)
        
        # Add nodes for each step
        workflow.add_node("initial_analysis", self.analyze_document)
        workflow.add_node("compliance_check", self.check_compliance)
        workflow.add_node("wait_for_human_review", self.wait_for_human_review)
        workflow.add_node("process_human_decision", self.process_human_decision)
        workflow.add_node("generate_revisions", self.generate_revisions)
        workflow.add_node("legal_escalation", self.escalate_to_legal)
        workflow.add_node("final_approval", self.final_approval)
        workflow.add_node("document_rejected", self.document_rejected)
        
        # Define the workflow edges
        workflow.add_edge(START, "initial_analysis")
        workflow.add_edge("initial_analysis", "compliance_check")
        
        # After compliance check, always require human review
        workflow.add_edge("compliance_check", "wait_for_human_review")
        
        # Conditional edges based on human decision
        workflow.add_conditional_edges(
            "wait_for_human_review",
            self.route_after_human_input,
            {
                "process_decision": "process_human_decision",
                "wait": "wait_for_human_review"  # Keep waiting
            }
        )
        
        workflow.add_conditional_edges(
            "process_human_decision",
            self.route_human_decision,
            {
                "approve": "final_approval",
                "reject": "document_rejected",
                "revise": "generate_revisions",
                "escalate": "legal_escalation"
            }
        )
        
        workflow.add_edge("generate_revisions", "wait_for_human_review")  # Back to human review
        workflow.add_edge("legal_escalation", "wait_for_human_review")    # Back to human review
        workflow.add_edge("final_approval", END)
        workflow.add_edge("document_rejected", END)
        
        return workflow.compile(checkpointer=MemorySaver())
    
    def _get_structured_response(self, prompt: str, response_model: BaseModel, max_retries: int = 2):
        """Get structured response using Pydantic model with retry logic."""
        for attempt in range(max_retries + 1):
            try:
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
                    return response_model.model_validate(parsed_data)
                else:
                    raise ValueError("No valid JSON found in response")
                    
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                print(f"⚠️ Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries:
                    # Return a default "failed" response based on model type
                    if response_model == DocumentAnalysis:
                        return DocumentAnalysis(
                            document_type="unknown",
                            risk_level=RiskLevel.HIGH,
                            confidence_score=0.0,
                            summary="Analysis failed - requires manual review",
                            requires_human_review=True
                        )
                    elif response_model == ComplianceCheck:
                        return ComplianceCheck(
                            is_compliant=False,
                            risk_score=10.0,
                            recommendations=["Manual compliance review required"],
                            requires_legal_review=True
                        )
                
                time.sleep(1)
    
    def analyze_document(self, state: DocumentWorkflowState) -> DocumentWorkflowState:
        """Node: AI analyzes the document for key terms and risks"""
        print("🔍 Analyzing document...")
        
        document_content = state["document_content"]
        document_metadata = state["document_metadata"]
        
        analysis_prompt = f"""
        Analyze this {document_metadata.get('type', 'document')} for potential legal and business risks:
        
        DOCUMENT CONTENT:
        {document_content[:2000]}...  # Truncate for API limits
        
        DOCUMENT INFO:
        Type: {document_metadata.get('type', 'Unknown')}
        Category: {document_metadata.get('category', 'Unknown')}
        
        Provide analysis including:
        1. Document type classification
        2. Key contractual terms identified
        3. Potential legal/business issues
        4. Compliance concerns
        5. Overall risk level (low/medium/high/critical)
        6. Confidence in analysis (0.0-1.0)
        7. Brief summary
        8. Whether human review is required
        """
        
        analysis_result = self._get_structured_response(analysis_prompt, DocumentAnalysis)
        
        # Add to audit trail
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": "ai_analysis",
            "action": "document_analyzed",
            "details": f"Risk level: {analysis_result.risk_level}, Confidence: {analysis_result.confidence_score}"
        }
        
        return {
            **state,
            "ai_analysis": analysis_result.model_dump(),
            "current_step": "initial_analysis",
            "audit_trail": state.get("audit_trail", []) + [audit_entry],
            "requires_human_input": analysis_result.requires_human_review
        }
    
    def check_compliance(self, state: DocumentWorkflowState) -> DocumentWorkflowState:
        """Node: Check document against compliance requirements"""
        print("⚖️ Checking compliance...")
        
        document_content = state["document_content"]
        ai_analysis = state["ai_analysis"]
        document_type = document_content[:500]  # First part for context
        
        compliance_prompt = f"""
        Check this document for compliance with relevant regulations:
        
        DOCUMENT TYPE: {ai_analysis.get('document_type', 'Unknown')}
        IDENTIFIED ISSUES: {ai_analysis.get('potential_issues', [])}
        
        DOCUMENT EXCERPT:
        {document_type}
        
        Check compliance with:
        1. General contract law requirements
        2. Data protection regulations (GDPR, etc.)
        3. Industry-specific regulations
        4. Corporate governance requirements
        
        Provide:
        1. List of regulations checked
        2. Any compliance violations found
        3. Overall compliance status (true/false)
        4. Risk score (0-10, where 10 is highest risk)
        5. Recommendations for addressing issues
        6. Whether legal expert review is required
        """
        
        compliance_result = self._get_structured_response(compliance_prompt, ComplianceCheck)
        
        # Add to audit trail
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": "compliance_check",
            "action": "compliance_verified",
            "details": f"Compliant: {compliance_result.is_compliant}, Risk score: {compliance_result.risk_score}"
        }
        
        return {
            **state,
            "compliance_results": compliance_result.model_dump(),
            "current_step": "compliance_check",
            "audit_trail": state.get("audit_trail", []) + [audit_entry],
            "requires_human_input": True  # Always require human review after compliance
        }
    
    def wait_for_human_review(self, state: DocumentWorkflowState) -> DocumentWorkflowState:
        """Node: Wait for human reviewer input - THIS IS THE HUMAN-IN-THE-LOOP CHECKPOINT"""
        print("⏳ Waiting for human review...")
        
        # Display current analysis for human reviewer
        ai_analysis = state.get("ai_analysis", {})
        compliance_results = state.get("compliance_results", {})
        
        print("\n" + "="*60)
        print("🧑‍💼 HUMAN REVIEW REQUIRED")
        print("="*60)
        print(f"📋 Document Type: {ai_analysis.get('document_type', 'Unknown')}")
        print(f"⚠️ Risk Level: {ai_analysis.get('risk_level', 'Unknown')}")
        print(f"✅ Compliance Status: {'Compliant' if compliance_results.get('is_compliant') else 'Non-Compliant'}")
        print(f"📊 Risk Score: {compliance_results.get('risk_score', 'Unknown')}/10")
        
        if ai_analysis.get('potential_issues'):
            print(f"🚨 Potential Issues:")
            for issue in ai_analysis.get('potential_issues', []):
                print(f"   • {issue}")
        
        if compliance_results.get('compliance_violations'):
            print(f"⚖️ Compliance Violations:")
            for violation in compliance_results.get('compliance_violations', []):
                print(f"   • {violation}")
        
        print("\n💡 AI Summary:", ai_analysis.get('summary', 'No summary available'))
        print("\n📝 Recommendations:")
        for rec in compliance_results.get('recommendations', []):
            print(f"   • {rec}")
        
        print("\n" + "="*60)
        print("HUMAN REVIEWER DECISION REQUIRED:")
        print("1. APPROVE - Document is acceptable as-is")
        print("2. REJECT - Document has unacceptable risks")
        print("3. REQUEST_REVISION - Document needs changes")
        print("4. ESCALATE - Needs higher-level legal review")
        print("="*60)
        
        # In a real system, this would wait for actual human input
        # For demo, we'll simulate different scenarios based on risk level
        return self._simulate_human_input(state)
    
    def _simulate_human_input(self, state: DocumentWorkflowState) -> DocumentWorkflowState:
        """Simulate human reviewer input based on document risk level"""
        
        # In production, this would be replaced with actual human input interface
        ai_analysis = state.get("ai_analysis", {})
        compliance_results = state.get("compliance_results", {})
        
        risk_level = ai_analysis.get('risk_level', 'high')
        is_compliant = compliance_results.get('is_compliant', False)
        risk_score = compliance_results.get('risk_score', 5.0)
        
        print("🤖 [SIMULATION] Generating human reviewer decision...")
        time.sleep(2)  # Simulate thinking time
        
        # Simulate different reviewer decisions based on risk
        if risk_level == 'low' and is_compliant and risk_score <= 3:
            decision = HumanDecision.APPROVE
            comments = "Low risk document with no compliance issues. Approved for execution."
            reviewer_role = "Legal Associate"
        elif risk_level == 'critical' or risk_score >= 8:
            decision = HumanDecision.ESCALATE
            comments = "High risk document requiring senior legal review before proceeding."
            reviewer_role = "Senior Legal Counsel"
        elif not is_compliant or risk_score >= 6:
            decision = HumanDecision.REQUEST_REVISION
            comments = "Document has compliance concerns that need to be addressed."
            reviewer_role = "Compliance Officer"
        elif risk_level == 'medium' and len(ai_analysis.get('potential_issues', [])) > 3:
            decision = HumanDecision.REQUEST_REVISION
            comments = "Multiple issues identified that require revision before approval."
            reviewer_role = "Contract Manager"
        else:
            decision = HumanDecision.APPROVE
            comments = "Document acceptable with minor risks. Approved with monitoring."
            reviewer_role = "Legal Associate"
        
        human_review = HumanReviewInput(
            reviewer_name=f"Reviewer_{len(state.get('human_reviews', [])) + 1}",
            reviewer_role=reviewer_role,
            decision=decision,
            comments=comments,
            additional_requirements=[] if decision == HumanDecision.APPROVE else ["Address compliance issues"],
            escalation_reason="High risk or compliance violation" if decision == HumanDecision.ESCALATE else None,
            timestamp=datetime.now().isoformat()
        )
        
        print(f"👤 Human Reviewer ({reviewer_role}) Decision: {decision.value.upper()}")
        print(f"💬 Comments: {comments}")
        
        return {
            **state,
            "human_reviews": state.get("human_reviews", []) + [human_review.model_dump()],
            "human_input_received": True,
            "current_step": "human_review_complete"
        }
    
    def route_after_human_input(self, state: DocumentWorkflowState) -> str:
        """Route based on whether human input has been received"""
        if state.get("human_input_received", False):
            return "process_decision"
        return "wait"
    
    def process_human_decision(self, state: DocumentWorkflowState) -> DocumentWorkflowState:
        """Node: Process the human reviewer's decision"""
        print("🔄 Processing human decision...")
        
        latest_review = state["human_reviews"][-1]  # Most recent review
        decision = latest_review["decision"]
        
        # Add to audit trail
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": "human_decision",
            "action": f"reviewer_decision_{decision}",
            "reviewer": latest_review["reviewer_name"],
            "details": latest_review["comments"]
        }
        
        return {
            **state,
            "current_step": "process_human_decision",
            "audit_trail": state.get("audit_trail", []) + [audit_entry],
            "human_input_received": False,  # Reset for potential next review
            "workflow_status": f"human_decided_{decision}"
        }
    
    def route_human_decision(self, state: DocumentWorkflowState) -> str:
        """Route based on human decision"""
        latest_review = state["human_reviews"][-1]
        decision = latest_review["decision"]
        
        routing_map = {
            "approve": "approve",
            "reject": "reject", 
            "request_revision": "revise",
            "escalate": "escalate"
        }
        
        return routing_map.get(decision, "escalate")
    
    def generate_revisions(self, state: DocumentWorkflowState) -> DocumentWorkflowState:
        """Node: Generate document revisions based on feedback"""
        print("📝 Generating revisions...")
        
        latest_review = state["human_reviews"][-1]
        ai_analysis = state.get("ai_analysis", {})
        compliance_results = state.get("compliance_results", {})
        
        revision_prompt = f"""
        Generate specific revision recommendations for this document based on the review feedback:
        
        REVIEWER FEEDBACK: {latest_review["comments"]}
        REVIEWER REQUIREMENTS: {latest_review.get("additional_requirements", [])}
        
        IDENTIFIED ISSUES: {ai_analysis.get("potential_issues", [])}
        COMPLIANCE VIOLATIONS: {compliance_results.get("compliance_violations", [])}
        
        Provide:
        1. Specific changes needed (requested_changes)
        2. Priority level for these changes
        3. Detailed instructions for implementation
        4. Optional deadline for completion
        """
        
        revision_request = RevisionRequest(
            requested_changes=ai_analysis.get("potential_issues", []) + compliance_results.get("compliance_violations", []),
            priority_level=RiskLevel.HIGH if latest_review["decision"] == "escalate" else RiskLevel.MEDIUM,
            specific_instructions=latest_review["comments"],
            deadline=None
        )
        
        print(f"📋 Revision Request Generated:")
        print(f"   Priority: {revision_request.priority_level}")
        print(f"   Changes needed: {len(revision_request.requested_changes)}")
        
        return {
            **state,
            "revision_requests": state.get("revision_requests", []) + [revision_request.model_dump()],
            "current_step": "revisions_generated",
            "requires_human_input": True  # Will need another human review after revisions
        }
    
    def escalate_to_legal(self, state: DocumentWorkflowState) -> DocumentWorkflowState:
        """Node: Escalate to senior legal team"""
        print("🚨 Escalating to senior legal review...")
        
        latest_review = state["human_reviews"][-1]
        
        print(f"📤 Document escalated to senior legal team")
        print(f"🔍 Escalation reason: {latest_review.get('escalation_reason', 'High risk document')}")
        print(f"⏰ Escalation timestamp: {datetime.now().isoformat()}")
        
        # Add to audit trail
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": "legal_escalation",
            "action": "escalated_to_senior_legal",
            "escalated_by": latest_review["reviewer_name"],
            "reason": latest_review.get("escalation_reason", "High risk")
        }
        
        return {
            **state,
            "current_step": "escalated",
            "workflow_status": "pending_senior_review",
            "audit_trail": state.get("audit_trail", []) + [audit_entry],
            "requires_human_input": True  # Senior legal will need to review
        }
    
    def final_approval(self, state: DocumentWorkflowState) -> DocumentWorkflowState:
        """Node: Document approved and finalized"""
        print("✅ Document approved!")
        
        latest_review = state["human_reviews"][-1]
        
        # Add to audit trail
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": "final_approval",
            "action": "document_approved",
            "approved_by": latest_review["reviewer_name"],
            "final_status": "APPROVED"
        }
        
        return {
            **state,
            "current_step": "approved",
            "workflow_status": "completed_approved",
            "final_decision": "APPROVED",
            "audit_trail": state.get("audit_trail", []) + [audit_entry]
        }
    
    def document_rejected(self, state: DocumentWorkflowState) -> DocumentWorkflowState:
        """Node: Document rejected"""
        print("❌ Document rejected!")
        
        latest_review = state["human_reviews"][-1]
        
        # Add to audit trail
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": "document_rejected",
            "action": "document_rejected",
            "rejected_by": latest_review["reviewer_name"],
            "rejection_reason": latest_review["comments"],
            "final_status": "REJECTED"
        }
        
        return {
            **state,
            "current_step": "rejected",
            "workflow_status": "completed_rejected",
            "final_decision": "REJECTED",
            "audit_trail": state.get("audit_trail", []) + [audit_entry]
        }
    
    def run(self, document_content: str, document_metadata: Dict) -> Dict:
        """Run the complete human-in-the-loop workflow"""
        
        initial_state = {
            "messages": [],
            "document_content": document_content,
            "document_metadata": document_metadata,
            "ai_analysis": None,
            "compliance_results": None,
            "human_reviews": [],
            "revision_requests": [],
            "current_step": "starting",
            "workflow_status": "initiated",
            "requires_human_input": False,
            "human_input_received": False,
            "retry_count": 0,
            "final_decision": None,
            "audit_trail": [{
                "timestamp": datetime.now().isoformat(),
                "step": "workflow_start",
                "action": "workflow_initiated",
                "details": f"Document type: {document_metadata.get('type', 'Unknown')}"
            }]
        }
        
        # Execute the workflow
        config = {"configurable": {"thread_id": f"doc_review_{int(time.time())}"}}
        final_state = self.graph.invoke(initial_state, config)
        
        return {
            "success": final_state["workflow_status"].startswith("completed"),
            "final_decision": final_state.get("final_decision"),
            "ai_analysis": final_state.get("ai_analysis"),
            "compliance_results": final_state.get("compliance_results"),
            "human_reviews": final_state.get("human_reviews", []),
            "revision_requests": final_state.get("revision_requests", []),
            "audit_trail": final_state.get("audit_trail", []),
            "workflow_state": final_state
        }

# Initialize the actual LLM client
def create_llm_client():
    """Create and return a Gemini LLM client"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,  # Lower temperature for more consistent analysis
        timeout=30,
        max_retries=2
    )

# Example usage
def main():
    """Example usage of the human-in-the-loop document review workflow"""
    
    # Initialize the workflow
    llm_client = create_llm_client()
    document_workflow = HumanInLoopDocumentWorkflow(llm_client)
    
    # Sample document content (abbreviated for demo)
    document_content = """
    SOFTWARE LICENSE AGREEMENT
    
    This Software License Agreement ("Agreement") is entered into between TechCorp Inc. 
    and the Licensee. The software includes proprietary algorithms and user data collection 
    capabilities.
    
    TERMS:
    1. Licensee agrees to unlimited data collection from end users
    2. All user data becomes property of TechCorp Inc.
    3. No warranty provided for software functionality
    4. Licensee liable for any damages caused by software
    5. Agreement governed by laws of jurisdiction favorable to TechCorp
    6. Automatic renewal with price increases at TechCorp's discretion
    
    DATA HANDLING:
    - Personal information including location, contacts, browsing history collected
    - Data shared with third-party partners without user consent
    - No user right to data deletion or correction
    
    This agreement supersedes all previous agreements and cannot be modified.
    """
    
    document_metadata = {
        "type": "Software License Agreement",
        "category": "Commercial Contract",
        "submitter": "Legal Department",
        "urgency": "Medium",
        "estimated_value": "$500,000"
    }
    
    print("🚀 Starting Human-in-the-Loop Document Review Workflow...")
    print("📄 Document: Software License Agreement")
    print("⚠️ Note: This document contains several problematic clauses for demo purposes")
    print("=" * 80)
    
    try:
        # Run the workflow
        results = document_workflow.run(document_content, document_metadata)
        
        print("\n" + "=" * 80)
        print("📊 WORKFLOW RESULTS")
        print("=" * 80)
        
        print(f"✅ Workflow Status: {results['success']}")
        print(f"🏁 Final Decision: {results.get('final_decision', 'PENDING')}")
        
        print(f"\n🔍 AI Analysis Summary:")
        ai_analysis = results.get("ai_analysis", {})
        print(f"   Document Type: {ai_analysis.get('document_type', 'Unknown')}")
        print(f"   Risk Level: {ai_analysis.get('risk_level', 'Unknown')}")
        print(f"   Issues Found: {len(ai_analysis.get('potential_issues', []))}")
        
        print(f"\n⚖️ Compliance Results:")
        compliance = results.get("compliance_results", {})
        print(f"   Compliant: {compliance.get('is_compliant', 'Unknown')}")
        print(f"   Risk Score: {compliance.get('risk_score', 'Unknown')}/10")
        print(f"   Violations: {len(compliance.get('compliance_violations', []))}")
        
        print(f"\n👥 Human Reviews: {len(results.get('human_reviews', []))}")
        for i, review in enumerate(results.get("human_reviews", []), 1):
            print(f"   Review {i}: {review['decision']} by {review['reviewer_role']}")
            print(f"   Comment: {review['comments']}")
        
        if results.get("revision_requests"):
            print(f"\n📝 Revision Requests: {len(results['revision_requests'])}")
            for req in results["revision_requests"]:
                print(f"   Priority: {req['priority_level']}")
                print(f"   Changes: {len(req['requested_changes'])}")
        
        print(f"\n📋 Audit Trail: {len(results.get('audit_trail', []))} entries")
        print("   Timeline:")
        for entry in results.get("audit_trail", []):
            timestamp = entry['timestamp'][:19]  # Remove microseconds
            print(f"   {timestamp}: {entry['action']} ({entry['step']})")
            
    except Exception as e:
        print(f"❌ Error running workflow: {str(e)}")
        print("Make sure you have set up your GOOGLE_API_KEY in your .env file")

if __name__ == "__main__":
    main()