"""
Adaptive RAG Example 1: Legal Research Assistant
This system handles both historical legal precedents and current legal developments
"""

import os
from typing import Literal, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict

from dotenv import load_dotenv
load_dotenv()

# Set up environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

class LegalQueryRouter(BaseModel):
    """Route legal queries to appropriate data sources."""
    
    datasource: Literal["legal_database", "current_legal_news", "case_law_search"] = Field(
        description="Choose data source: legal_database for established law, current_legal_news for recent developments, case_law_search for precedents"
    )
    
    complexity: Literal["simple", "complex", "multi_jurisdictional"] = Field(
        description="Assess query complexity to determine retrieval depth"
    )

class LegalDocumentGrader(BaseModel):
    """Grade legal documents for relevance and authority."""
    
    relevance_score: str = Field(description="Document relevance: 'high', 'medium', 'low'")
    authority_level: str = Field(description="Legal authority: 'binding', 'persuasive', 'commentary'")
    jurisdiction_match: str = Field(description="Jurisdiction relevance: 'exact', 'similar', 'different'")

class LegalGraphState(TypedDict):
    question: str
    legal_context: str
    jurisdiction: str
    documents: List[str]
    generation: str
    complexity_level: str

class LegalAdaptiveRAG:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.web_search = TavilySearchResults(k=3)
        
        # Initialize legal database (case law, statutes, regulations)
        self.legal_vectorstore = self._build_legal_database()
        
        # Set up routers and graders
        self.query_router = self._setup_query_router()
        self.document_grader = self._setup_document_grader()
        
    def _build_legal_database(self):
        """Build vectorstore with legal documents."""
        # Load legal documents (PDFs of case law, statutes, etc.)
        legal_docs = [
            "constitutional_law_cases.pdf",
            "contract_law_statutes.pdf",
            "tort_law_precedents.pdf"
        ]
        
        documents = []
        for doc_path in legal_docs:
            if os.path.exists(doc_path):
                loader = PyPDFLoader(doc_path)
                documents.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        if documents:
            doc_splits = text_splitter.split_documents(documents)
            vectorstore = FAISS.from_documents(doc_splits, self.embeddings)
            return vectorstore.as_retriever()
        return None
    
    def _setup_query_router(self):
        """Set up intelligent query routing for legal queries."""
        router_llm = self.llm.with_structured_output(LegalQueryRouter)
        
        system_prompt = """You are an expert legal research assistant. Analyze the query and determine:
        1. The appropriate data source (legal_database for established law, current_legal_news for recent developments, case_law_search for precedents)
        2. The complexity level (simple for straightforward questions, complex for multi-faceted issues, multi_jurisdictional for cross-border matters)
        
        Consider factors like:
        - Recency of the legal issue
        - Jurisdiction specificity
        - Complexity of legal concepts involved
        - Need for current vs. historical information
        """
        
        route_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Legal Query: {question}\nJurisdiction: {jurisdiction}")
        ])
        
        return route_prompt | router_llm
    
    def _setup_document_grader(self):
        """Set up legal document grading system."""
        grader_llm = self.llm.with_structured_output(LegalDocumentGrader)
        
        system_prompt = """You are a legal document evaluator. Assess documents based on:
        1. Relevance to the legal question
        2. Authority level (binding precedent, persuasive authority, or commentary)
        3. Jurisdiction match (exact, similar, or different jurisdiction)
        
        Prioritize binding precedents from the same jurisdiction over persuasive authorities.
        """
        
        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Document: {document}\nLegal Question: {question}\nJurisdiction: {jurisdiction}")
        ])
        
        return grade_prompt | grader_llm
    
    def route_legal_query(self, state):
        """Route legal queries based on type and complexity."""
        print("---ROUTING LEGAL QUERY---")
        
        question = state["question"]
        jurisdiction = state.get("jurisdiction", "Federal")
        
        routing_decision = self.query_router.invoke({
            "question": question,
            "jurisdiction": jurisdiction
        })
        
        print(f"Routing to: {routing_decision.datasource}")
        print(f"Complexity: {routing_decision.complexity}")
        
        return {
            **state,
            "complexity_level": routing_decision.complexity
        }
    
    def retrieve_legal_documents(self, state):
        """Retrieve relevant legal documents based on routing decision."""
        print("---RETRIEVING LEGAL DOCUMENTS---")
        
        question = state["question"]
        complexity = state["complexity_level"]
        
        if complexity == "simple":
            # Simple retrieval from legal database
            if self.legal_vectorstore:
                docs = self.legal_vectorstore.invoke(question)
                return {**state, "documents": docs[:3]}
        
        elif complexity == "complex":
            # Multi-source retrieval
            legal_docs = []
            if self.legal_vectorstore:
                legal_docs = self.legal_vectorstore.invoke(question)
            
            # Also search for recent legal developments
            web_results = self.web_search.invoke({"query": f"recent legal developments {question}"})
            
            # Combine and prioritize results
            all_docs = legal_docs[:2] + [{"content": result["content"]} for result in web_results]
            return {**state, "documents": all_docs}
        
        else:  # multi_jurisdictional
            # Comprehensive search across multiple jurisdictions
            jurisdictions = ["Federal", "State", "International"]
            all_docs = []
            
            for jurisdiction in jurisdictions:
                query = f"{question} {jurisdiction} law"
                if self.legal_vectorstore:
                    docs = self.legal_vectorstore.invoke(query)
                    all_docs.extend(docs[:2])
            
            return {**state, "documents": all_docs}
    
    def grade_legal_documents(self, state):
        """Grade legal documents for authority and relevance."""
        print("---GRADING LEGAL DOCUMENTS---")
        
        question = state["question"]
        documents = state["documents"]
        jurisdiction = state.get("jurisdiction", "Federal")
        
        graded_docs = []
        
        for doc in documents:
            doc_content = doc.get("content", "") if isinstance(doc, dict) else str(doc)
            
            grade = self.document_grader.invoke({
                "document": doc_content,
                "question": question,
                "jurisdiction": jurisdiction
            })
            
            # Prioritize based on authority and relevance
            if grade.authority_level == "binding" and grade.relevance_score in ["high", "medium"]:
                graded_docs.append(doc)
            elif grade.authority_level == "persuasive" and grade.relevance_score == "high":
                graded_docs.append(doc)
        
        print(f"Filtered to {len(graded_docs)} relevant legal documents")
        return {**state, "documents": graded_docs}
    
    def generate_legal_response(self, state):
        """Generate legal response with proper citations and disclaimers."""
        print("---GENERATING LEGAL RESPONSE---")
        
        question = state["question"]
        documents = state["documents"]
        jurisdiction = state.get("jurisdiction", "Federal")
        
        legal_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a legal research assistant. Provide a comprehensive legal analysis based on the retrieved documents.
            
            Format your response with:
            1. Legal Analysis
            2. Relevant Case Law/Statutes
            3. Jurisdictional Considerations
            4. Practical Implications
            5. Disclaimer
            
            Always include proper legal citations and note the jurisdiction.
            Add a disclaimer that this is for informational purposes only."""),
            ("human", """Question: {question}
            Jurisdiction: {jurisdiction}
            Legal Documents: {documents}
            
            Provide a detailed legal analysis.""")
        ])
        
        legal_chain = legal_prompt | self.llm
        
        response = legal_chain.invoke({
            "question": question,
            "jurisdiction": jurisdiction,
            "documents": documents
        })
        
        return {**state, "generation": response.content}

# Example Usage
def run_legal_example():
    """Demonstrate legal research with different complexity levels."""
    
    legal_rag = LegalAdaptiveRAG()
    
    # Test cases with different complexity levels
    test_cases = [
        {
            "question": "What are the basic elements of a contract?",
            "jurisdiction": "Federal",
            "expected_complexity": "simple"
        },
        {
            "question": "How does the recent Supreme Court decision on digital privacy affect corporate data collection policies?",
            "jurisdiction": "Federal",
            "expected_complexity": "complex"
        },
        {
            "question": "What are the conflict of laws principles when a contract dispute involves parties from different countries?",
            "jurisdiction": "International",
            "expected_complexity": "multi_jurisdictional"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"LEGAL RESEARCH EXAMPLE {i}")
        print(f"{'='*50}")
        
        # Create initial state
        state = {
            "question": test_case["question"],
            "jurisdiction": test_case["jurisdiction"],
            "documents": [],
            "generation": "",
            "complexity_level": ""
        }
        
        # Process through pipeline
        state = legal_rag.route_legal_query(state)
        state = legal_rag.retrieve_legal_documents(state)
        state = legal_rag.grade_legal_documents(state)
        state = legal_rag.generate_legal_response(state)
        
        print(f"Final Legal Analysis:")
        print(state["generation"])
        print(f"Complexity Level: {state['complexity_level']}")

if __name__ == "__main__":
    run_legal_example()