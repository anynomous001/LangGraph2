"""
Customer Support Ticket Classification & Response System
Human-in-the-Loop Example

Real-world scenario: AI helps classify and draft responses for customer support tickets,
but human agents make the final decisions on important issues.
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass

# Simulate different types of customer issues
class TicketPriority(str, Enum):
    LOW = "low"           # FAQ, simple questions
    MEDIUM = "medium"     # Account issues, feature requests  
    HIGH = "high"         # Billing problems, service issues
    URGENT = "urgent"     # System outages, security issues

class TicketCategory(str, Enum):
    TECHNICAL = "technical"
    BILLING = "billing"
    ACCOUNT = "account"
    PRODUCT = "product"
    COMPLAINT = "complaint"

class HumanAction(str, Enum):
    APPROVE = "approve"           # Send AI response as-is
    EDIT_AND_SEND = "edit_send"   # Modify AI response before sending
    ESCALATE = "escalate"         # Send to senior agent
    REJECT = "reject"             # Write completely new response

@dataclass
class CustomerTicket:
    ticket_id: str
    customer_name: str
    email: str
    subject: str
    message: str
    timestamp: str
    customer_tier: str = "standard"  # standard, premium, enterprise

@dataclass
class AIAnalysis:
    priority: TicketPriority
    category: TicketCategory
    confidence: float  # 0.0 to 1.0
    suggested_response: str
    key_issues: List[str]
    requires_human_review: bool
    escalation_reason: Optional[str] = None

@dataclass
class HumanDecision:
    agent_name: str
    action: HumanAction
    confidence_in_ai: str  # "high", "medium", "low"
    edited_response: Optional[str] = None
    notes: str = ""
    time_spent_seconds: int = 0

class CustomerSupportHITL:
    """Simple Human-in-the-Loop Customer Support System"""
    
    def __init__(self):
        # These would normally be ML models
        self.knowledge_base = {
            "password reset": "To reset your password, click 'Forgot Password' on the login page and follow the email instructions.",
            "billing question": "For billing inquiries, please check your account dashboard or contact our billing team at billing@company.com",
            "technical issue": "We're sorry you're experiencing technical difficulties. Our team will investigate and get back to you within 24 hours.",
            "refund request": "Refund requests are processed within 5-7 business days. Please provide your order number for faster processing.",
            "account locked": "Your account appears to be temporarily locked for security. Please verify your identity to unlock it."
        }
        
        self.escalation_keywords = [
            "angry", "frustrated", "lawsuit", "attorney", "cancel", "refund", 
            "terrible", "worst", "hate", "discrimination", "legal action"
        ]
        
        self.urgent_keywords = [
            "urgent", "emergency", "down", "not working", "broken", "critical",
            "security breach", "hack", "unauthorized access"
        ]
    
    def classify_ticket(self, ticket: CustomerTicket) -> AIAnalysis:
        """AI Step 1: Classify the ticket and determine priority"""
        
        print(f"🤖 AI analyzing ticket #{ticket.ticket_id}...")
        
        # Simple keyword-based classification (in reality, this would be ML)
        text_to_analyze = f"{ticket.subject} {ticket.message}".lower()
        
        # Determine priority
        if any(keyword in text_to_analyze for keyword in self.urgent_keywords):
            priority = TicketPriority.URGENT
        elif any(keyword in text_to_analyze for keyword in self.escalation_keywords):
            priority = TicketPriority.HIGH
        elif "billing" in text_to_analyze or "payment" in text_to_analyze:
            priority = TicketPriority.HIGH
        elif "password" in text_to_analyze or "login" in text_to_analyze:
            priority = TicketPriority.LOW
        else:
            priority = TicketPriority.MEDIUM
        
        # Determine category
        if "password" in text_to_analyze or "login" in text_to_analyze or "technical" in text_to_analyze:
            category = TicketCategory.TECHNICAL
        elif "billing" in text_to_analyze or "payment" in text_to_analyze or "refund" in text_to_analyze:
            category = TicketCategory.BILLING
        elif "account" in text_to_analyze:
            category = TicketCategory.ACCOUNT
        elif any(keyword in text_to_analyze for keyword in self.escalation_keywords):
            category = TicketCategory.COMPLAINT
        else:
            category = TicketCategory.PRODUCT
        
        # Calculate confidence (simplified)
        confidence = 0.9 if priority == TicketPriority.LOW else 0.7
        
        # Generate suggested response
        suggested_response = self._generate_response(ticket, category, priority)
        
        # Determine if human review is needed
        requires_human_review = (
            priority in [TicketPriority.HIGH, TicketPriority.URGENT] or
            ticket.customer_tier in ["premium", "enterprise"] or
            any(keyword in text_to_analyze for keyword in self.escalation_keywords) or
            confidence < 0.8
        )
        
        escalation_reason = None
        if priority == TicketPriority.URGENT:
            escalation_reason = "Urgent priority requires immediate human attention"
        elif any(keyword in text_to_analyze for keyword in self.escalation_keywords):
            escalation_reason = "Customer appears frustrated - needs human touch"
        elif ticket.customer_tier in ["premium", "enterprise"]:
            escalation_reason = "High-value customer requires personalized response"
        
        key_issues = self._extract_key_issues(text_to_analyze)
        
        return AIAnalysis(
            priority=priority,
            category=category,
            confidence=confidence,
            suggested_response=suggested_response,
            key_issues=key_issues,
            requires_human_review=requires_human_review,
            escalation_reason=escalation_reason
        )
    
    def _generate_response(self, ticket: CustomerTicket, category: TicketCategory, priority: TicketPriority) -> str:
        """Generate AI response based on ticket analysis"""
        
        # Personalized greeting
        greeting = f"Dear {ticket.customer_name},"
        
        # Get template response based on category
        if "password" in ticket.message.lower():
            template = self.knowledge_base["password reset"]
        elif "billing" in ticket.message.lower():
            template = self.knowledge_base["billing question"]  
        elif "refund" in ticket.message.lower():
            template = self.knowledge_base["refund request"]
        elif "locked" in ticket.message.lower():
            template = self.knowledge_base["account locked"]
        else:
            template = self.knowledge_base["technical issue"]
        
        # Add priority-based urgency
        if priority == TicketPriority.URGENT:
            urgency = "This is marked as urgent and we're prioritizing your request."
        elif priority == TicketPriority.HIGH:
            urgency = "We understand this is important to you and will respond quickly."
        else:
            urgency = "Thank you for contacting us."
        
        # Closing
        closing = "Best regards,\nCustomer Support Team\nTicket #" + ticket.ticket_id
        
        return f"{greeting}\n\n{urgency}\n\n{template}\n\n{closing}"
    
    def _extract_key_issues(self, text: str) -> List[str]:
        """Extract key issues from ticket text"""
        issues = []
        
        if "password" in text:
            issues.append("Password/Login Issue")
        if "billing" in text or "payment" in text:
            issues.append("Billing/Payment Issue") 
        if "not working" in text or "broken" in text:
            issues.append("Technical Malfunction")
        if "refund" in text:
            issues.append("Refund Request")
        if any(word in text for word in self.escalation_keywords):
            issues.append("Customer Dissatisfaction")
        if "urgent" in text:
            issues.append("Urgent Request")
        
        return issues if issues else ["General Inquiry"]
    
    def wait_for_human_review(self, ticket: CustomerTicket, ai_analysis: AIAnalysis) -> HumanDecision:
        """Human-in-the-Loop: Get human agent decision"""
        
        print("\n" + "="*70)
        print("👤 HUMAN AGENT REVIEW REQUIRED")
        print("="*70)
        
        print(f"📧 Ticket: #{ticket.ticket_id}")
        print(f"👤 Customer: {ticket.customer_name} ({ticket.customer_tier})")
        print(f"📞 Email: {ticket.email}")
        print(f"📋 Subject: {ticket.subject}")
        print(f"💬 Message: {ticket.message[:200]}...")
        
        print(f"\n🤖 AI ANALYSIS:")
        print(f"   Priority: {ai_analysis.priority.value.upper()}")
        print(f"   Category: {ai_analysis.category.value}")
        print(f"   Confidence: {ai_analysis.confidence:.1%}")
        print(f"   Key Issues: {', '.join(ai_analysis.key_issues)}")
        if ai_analysis.escalation_reason:
            print(f"   ⚠️ Escalation Reason: {ai_analysis.escalation_reason}")
        
        print(f"\n📝 AI SUGGESTED RESPONSE:")
        print("-" * 50)
        print(ai_analysis.suggested_response)
        print("-" * 50)
        
        print(f"\n❓ WHAT SHOULD WE DO?")
        print("1. APPROVE - Send AI response as-is")
        print("2. EDIT_AND_SEND - Modify response before sending")  
        print("3. ESCALATE - Send to senior agent")
        print("4. REJECT - Write completely new response")
        
        # Simulate human decision based on AI analysis
        return self._simulate_human_decision(ticket, ai_analysis)
    
    def _simulate_human_decision(self, ticket: CustomerTicket, ai_analysis: AIAnalysis) -> HumanDecision:
        """Simulate human agent decision-making"""
        
        print("\n🤔 Agent thinking...")
        time.sleep(2)  # Simulate thinking time
        
        # Simulate different agent decisions based on various factors
        if ai_analysis.priority == TicketPriority.LOW and ai_analysis.confidence > 0.8:
            # Low priority, high confidence - likely to approve
            action = HumanAction.APPROVE
            confidence = "high"
            notes = "Straightforward issue, AI response looks good"
            agent_name = "Agent_Sarah"
            time_spent = 30
            
        elif ai_analysis.priority == TicketPriority.URGENT:
            # Urgent issues need escalation
            action = HumanAction.ESCALATE  
            confidence = "medium"
            notes = "Urgent issue requiring immediate senior agent attention"
            agent_name = "Agent_Mike" 
            time_spent = 45
            
        elif any("angry" in issue.lower() for issue in ai_analysis.key_issues) or "Customer Dissatisfaction" in ai_analysis.key_issues:
            # Angry customer needs human touch
            action = HumanAction.EDIT_AND_SEND
            confidence = "low"
            notes = "Customer seems frustrated, adding more empathy to response"
            agent_name = "Agent_Lisa"
            time_spent = 120
            edited_response = self._add_empathy_to_response(ai_analysis.suggested_response, ticket.customer_name)
            
        elif ticket.customer_tier in ["premium", "enterprise"]:
            # High-value customers get personalized responses
            action = HumanAction.EDIT_AND_SEND
            confidence = "medium" 
            notes = "Premium customer deserves personalized response"
            agent_name = "Agent_David"
            time_spent = 90
            edited_response = self._personalize_response(ai_analysis.suggested_response, ticket)
            
        elif ai_analysis.confidence < 0.7:
            # Low confidence AI responses need human review
            action = HumanAction.REJECT
            confidence = "low"
            notes = "AI confidence too low, writing custom response"
            agent_name = "Agent_Emma"
            time_spent = 180
            
        else:
            # Default case - edit and improve
            action = HumanAction.EDIT_AND_SEND
            confidence = "medium"
            notes = "Minor edits to improve tone and clarity"
            agent_name = "Agent_Alex"
            time_spent = 60
            edited_response = ai_analysis.suggested_response.replace("Customer Support Team", f"Customer Support Team\n(Agent: {agent_name})")
        
        decision = HumanDecision(
            agent_name=agent_name,
            action=action,
            confidence_in_ai=confidence,
            notes=notes,
            time_spent_seconds=time_spent
        )
        
        if action in [HumanAction.EDIT_AND_SEND] and 'edited_response' in locals():
            decision.edited_response = edited_response
        
        print(f"👤 Agent {agent_name} Decision: {action.value.upper()}")
        print(f"💭 Confidence in AI: {confidence}")
        print(f"📝 Notes: {notes}")
        print(f"⏱️ Time spent: {time_spent} seconds")
        
        return decision
    
    def _add_empathy_to_response(self, original_response: str, customer_name: str) -> str:
        """Add more empathetic language for frustrated customers"""
        
        lines = original_response.split('\n')
        # Replace the greeting with more empathetic one
        lines[0] = f"Dear {customer_name},"
        lines[2] = "I sincerely apologize for any frustration this issue has caused you. Your experience is very important to us, and I want to personally ensure we resolve this quickly."
        
        return '\n'.join(lines)
    
    def _personalize_response(self, original_response: str, ticket: CustomerTicket) -> str:
        """Personalize response for premium customers"""
        
        lines = original_response.split('\n')
        lines[0] = f"Dear {ticket.customer_name},"
        lines[2] = f"Thank you for being a valued {ticket.customer_tier} customer. I'm personally handling your request to ensure you receive the best possible service."
        
        # Add premium customer benefits
        lines.insert(-2, "As a premium customer, you also have access to our priority support line at 1-800-PREMIUM for any future urgent needs.")
        
        return '\n'.join(lines)
    
    def execute_decision(self, decision: HumanDecision, ai_analysis: AIAnalysis) -> str:
        """Execute the human agent's decision"""
        
        if decision.action == HumanAction.APPROVE:
            final_response = ai_analysis.suggested_response
            print("✅ AI response approved and sent!")
            
        elif decision.action == HumanAction.EDIT_AND_SEND:
            final_response = decision.edited_response or ai_analysis.suggested_response
            print("✏️ Response edited and sent!")
            
        elif decision.action == HumanAction.ESCALATE:
            final_response = f"Ticket escalated to senior agent. Response pending."
            print("🚨 Ticket escalated to senior agent!")
            
        elif decision.action == HumanAction.REJECT:
            final_response = "Custom human response written and sent."
            print("✍️ Custom response written by human agent!")
        
        return final_response
    
    def process_ticket(self, ticket: CustomerTicket) -> Dict:
        """Complete workflow: AI analysis + Human decision + Execution"""
        
        print(f"\n🎫 Processing ticket #{ticket.ticket_id} from {ticket.customer_name}")
        
        # Step 1: AI Analysis
        ai_analysis = self.classify_ticket(ticket)
        
        # Step 2: Human Review (if required)
        if ai_analysis.requires_human_review:
            human_decision = self.wait_for_human_review(ticket, ai_analysis)
            
            # Step 3: Execute Decision
            final_response = self.execute_decision(human_decision, ai_analysis)
        else:
            # Auto-approve low-risk tickets
            print("✅ Low-risk ticket auto-approved!")
            human_decision = HumanDecision(
                agent_name="AI_Auto_Approve",
                action=HumanAction.APPROVE,
                confidence_in_ai="high",
                notes="Low risk ticket auto-approved",
                time_spent_seconds=5
            )
            final_response = ai_analysis.suggested_response
        
        return {
            "ticket_id": ticket.ticket_id,
            "ai_analysis": ai_analysis,
            "human_decision": human_decision,
            "final_response": final_response,
            "timestamp": datetime.now().isoformat()
        }

def main():
    """Demo the customer support human-in-the-loop system"""
    
    system = CustomerSupportHITL()
    
    # Sample tickets demonstrating different scenarios
    sample_tickets = [
        CustomerTicket(
            ticket_id="T001",
            customer_name="John Smith", 
            email="john@email.com",
            subject="Password Reset Help",
            message="Hi, I forgot my password and can't log in. Can you help me reset it?",
            timestamp=datetime.now().isoformat(),
            customer_tier="standard"
        ),
        
        CustomerTicket(
            ticket_id="T002",
            customer_name="Sarah Johnson",
            email="sarah@bigcorp.com", 
            subject="URGENT: System completely down!",
            message="This is absolutely critical! Our entire system is down and we're losing money every minute. This is unacceptable! We need this fixed IMMEDIATELY or we're considering legal action!",
            timestamp=datetime.now().isoformat(),
            customer_tier="enterprise"
        ),
        
        CustomerTicket(
            ticket_id="T003",
            customer_name="Mike Davis",
            email="mike@startup.com",
            subject="Billing Question",
            message="I was charged twice this month and I'm really frustrated. This is the third billing error in 6 months. I'm considering canceling my subscription.",
            timestamp=datetime.now().isoformat(),
            customer_tier="premium"
        )
    ]
    
    print("🏢 Customer Support - Human-in-the-Loop Demo")
    print("="*80)
    print("This system shows how AI helps classify and draft responses,")
    print("but human agents make final decisions on important tickets.")
    print("="*80)
    
    results = []
    
    for ticket in sample_tickets:
        result = system.process_ticket(ticket)
        results.append(result)
        print("\n" + "="*80)
    
    # Summary
    print("\n📊 WORKFLOW SUMMARY")
    print("="*80)
    
    total_human_time = sum(r["human_decision"].time_spent_seconds for r in results)
    auto_approved = sum(1 for r in results if r["human_decision"].action == HumanAction.APPROVE)
    escalated = sum(1 for r in results if r["human_decision"].action == HumanAction.ESCALATE)
    
    print(f"📧 Total Tickets Processed: {len(results)}")
    print(f"⚡ Auto-approved: {auto_approved}")
    print(f"🚨 Escalated: {escalated}")
    print(f"⏱️ Total Human Time: {total_human_time} seconds ({total_human_time/60:.1f} minutes)")
    print(f"💡 Average Time per Ticket: {total_human_time/len(results):.0f} seconds")
    
    print("\n🎯 Key Benefits:")
    print("• AI handles initial classification and draft responses")
    print("• Humans focus on high-priority and complex issues")
    print("• Customers get faster, more accurate responses")
    print("• Quality is maintained through human oversight")
    
if __name__ == "__main__":
    main()