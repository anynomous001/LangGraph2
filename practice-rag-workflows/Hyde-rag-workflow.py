"""
HyDE RAG: Medical Research Assistant
Real-world use case: Helping oncologists and researchers find relevant medical literature
by first generating hypothetical ideal research papers, then using them to guide retrieval.

Use Case: A medical researcher asks: "What are the latest immunotherapy approaches for treating 
pediatric acute lymphoblastic leukemia with resistance to conventional chemotherapy?"

Instead of directly searching with this query, HyDE first generates what an ideal research paper
would look like, then uses that hypothetical document to find actual relevant papers.
"""

import os
import json
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import numpy as np
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq     



from dotenv import load_dotenv
load_dotenv()

# Set up environment variables
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")


class MedicalQuery(BaseModel):
    """Structure for medical research queries."""
    
    condition: str = Field(description="Primary medical condition or disease")
    treatment_type: str = Field(description="Type of treatment being researched")
    patient_population: str = Field(description="Target patient demographics")
    urgency: str = Field(description="Research urgency: 'routine', 'urgent', 'critical'")

class HyDEMedicalState(TypedDict):
    """State management for HyDE medical research workflow."""
    
    original_query: str
    parsed_query: MedicalQuery
    hypothetical_document: str
    retrieved_documents: List[Dict[str, Any]]
    final_response: str
    search_embeddings: List[float]

class HyDEMedicalRAG:
    """
    HyDE RAG system specifically designed for medical research queries.
    
    This system is particularly valuable for medical research because:
    1. Medical queries are often complex and multi-faceted
    2. Ideal research papers have specific structures (abstract, methods, results, conclusions)
    3. Medical terminology can be domain-specific and benefit from hypothetical context
    """
    
    def __init__(self):
        self.llm =ChatGroq(model="Gemma2-9b-It")
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.getenv("GOOGLE_API_KEY"))
        self.web_search = TavilySearch(k=5)
        
        # Build medical knowledge base
        self.medical_vectorstore = self._build_medical_knowledge_base()
        
        # Initialize prompts
        self._setup_prompts()
    
    def _build_medical_knowledge_base(self):
        """
        Build a medical knowledge base with sample research papers.
        In a real implementation, this would contain thousands of medical papers.
        """
        print("📚 Building medical knowledge base...")
        
        # Sample medical research papers content
        sample_papers = [
            {
                "title": "CAR-T Cell Therapy for Pediatric Acute Lymphoblastic Leukemia: Recent Advances",
                "content": """
                Abstract: Chimeric Antigen Receptor T-cell (CAR-T) therapy has revolutionized treatment 
                for pediatric acute lymphoblastic leukemia (ALL). This study reviews recent advances in 
                CAR-T cell therapy, including novel antigen targets, enhanced safety profiles, and 
                improved persistence. Methods: We analyzed 156 pediatric patients treated with 
                tisagenlecleucel between 2019-2024. Results: Complete remission rates of 89% were 
                achieved in relapsed/refractory B-cell ALL patients. Conclusion: CAR-T therapy 
                represents a paradigm shift in pediatric oncology with continued improvements in 
                efficacy and safety profiles.
                """,
                "authors": "Smith, J.A., et al.",
                "journal": "Journal of Pediatric Oncology",
                "year": "2024",
                "keywords": ["CAR-T", "pediatric", "ALL", "immunotherapy"]
            },
            {
                "title": "Blinatumomab in Chemotherapy-Resistant Pediatric B-ALL: A Multicenter Study",
                "content": """
                Abstract: Blinatumomab, a bispecific T-cell engager (BiTE), has shown promising results 
                in pediatric patients with chemotherapy-resistant B-cell acute lymphoblastic leukemia. 
                This multicenter study evaluates safety and efficacy outcomes. Methods: 78 pediatric 
                patients with relapsed/refractory B-ALL received blinatumomab therapy. Results: 
                Overall response rate was 73% with manageable toxicity profile. Cytokine release 
                syndrome occurred in 23% of patients. Conclusion: Blinatumomab provides an effective 
                bridge to transplantation for high-risk pediatric ALL patients.
                """,
                "authors": "Johnson, M.B., et al.",
                "journal": "Blood Cancer Research",
                "year": "2024",
                "keywords": ["blinatumomab", "BiTE", "pediatric", "chemotherapy-resistant"]
            },
            {
                "title": "Novel Immunotherapeutic Approaches for Pediatric Hematological Malignancies",
                "content": """
                Abstract: The landscape of pediatric hematological malignancy treatment has evolved 
                significantly with novel immunotherapeutic approaches. This review examines emerging 
                strategies including checkpoint inhibitors, cellular therapies, and combination 
                approaches. Methods: Comprehensive literature review of immunotherapy trials in 
                pediatric patients from 2020-2024. Results: Combination immunotherapy strategies 
                show enhanced efficacy with acceptable toxicity profiles. Conclusion: 
                Immunotherapy represents the future of pediatric cancer treatment with personalized 
                approaches becoming increasingly viable.
                """,
                "authors": "Chen, L.K., et al.",
                "journal": "Pediatric Hematology Today",
                "year": "2024",
                "keywords": ["immunotherapy", "pediatric", "hematological", "malignancies"]
            }
        ]
        
        # Convert to documents and create embeddings
        documents = []
        for paper in sample_papers:
            doc_content = f"Title: {paper['title']}\nContent: {paper['content']}\nAuthors: {paper['authors']}\nJournal: {paper['journal']}\nYear: {paper['year']}\nKeywords: {', '.join(paper['keywords'])}"
            documents.append(doc_content)
        
        # Create FAISS vectorstore
        if documents:
            vectorstore = FAISS.from_texts(documents, self.embeddings)
            return vectorstore.as_retriever(search_kwargs={"k": 3})
        
        return None
    
    def _setup_prompts(self):
        """Initialize all the prompts used in the HyDE workflow."""
        
        # Step 1: Query Analysis Prompt
        self.query_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a medical research analyst. Parse the medical query and extract:
            1. Primary condition/disease
            2. Treatment type being researched
            3. Target patient population
            4. Research urgency level
            
            Return a structured analysis."""),
            ("human", "Medical Query: {query}")
        ])
        
        # Step 2: Hypothetical Document Generation Prompt
        self.hypothetical_doc_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a medical research expert. Generate a hypothetical research paper 
            that would perfectly answer the given medical query. The paper should include:
            
            1. A compelling title
            2. Abstract with background, methods, results, and conclusions
            3. Key findings that directly address the query
            4. Relevant medical terminology and concepts
            5. Specific data points and statistics (realistic but hypothetical)
            
            Make it sound like a real research paper that would be published in a top medical journal.
            Focus on the specific medical condition, treatment approach, and patient population mentioned."""),
            ("human", """Generate a hypothetical research paper for this medical query:
            
            Query: {query}
            Condition: {condition}
            Treatment: {treatment_type}
            Population: {patient_population}
            
            Create an ideal research paper that would perfectly answer this query.""")
        ])
        
        # Step 3: Final Response Generation Prompt  
        self.response_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a medical research assistant providing evidence-based responses 
            to healthcare professionals. Use the retrieved medical literature to provide:
            
            1. A comprehensive answer to the original query
            2. Current treatment approaches and their efficacy
            3. Recent research developments
            4. Clinical considerations and recommendations
            5. Limitations and areas for future research
            
            Always cite relevant studies and maintain scientific accuracy.
            Include appropriate medical disclaimers."""),
            ("human", """Original Query: {query}
            
            Retrieved Medical Literature:
            {documents}
            
            Provide a comprehensive, evidence-based response suitable for medical professionals.""")
        ])
    
    def step1_analyze_query(self, state: HyDEMedicalState) -> HyDEMedicalState:
        """
        STEP 1: QUERY ANALYSIS
        Parse the medical query to understand the specific medical context.
        This helps tailor the hypothetical document generation.
        """
        print("\n🔍 STEP 1: Analyzing medical query...")
        
        query = state["original_query"]
        
        # Use structured output to parse the query
        structured_llm = self.llm.with_structured_output(MedicalQuery)
        
        try:
            parsed_query = structured_llm.invoke([
                ("system", "Analyze this medical query and extract key components."),
                ("human", f"Query: {query}")
            ])
            
            print(f"   📋 Condition: {parsed_query.condition}")
            print(f"   💊 Treatment: {parsed_query.treatment_type}")
            print(f"   👥 Population: {parsed_query.patient_population}")
            print(f"   ⚡ Urgency: {parsed_query.urgency}")
            
            state["parsed_query"] = parsed_query
            
        except Exception as e:
            print(f"   ⚠️ Error in query analysis: {e}")
            # Fallback to basic parsing
            state["parsed_query"] = MedicalQuery(
                condition="general",
                treatment_type="various",
                patient_population="general",
                urgency="routine"
            )
        
        return state
    
    def step2_generate_hypothetical_document(self, state: HyDEMedicalState) -> HyDEMedicalState:
        """
        STEP 2: HYPOTHETICAL DOCUMENT GENERATION
        This is the core of HyDE - generate what an ideal research paper would look like.
        The hypothetical document serves as a semantic guide for retrieval.
        """
        print("\n📝 STEP 2: Generating hypothetical research paper...")
        
        query = state["original_query"]
        parsed_query = state["parsed_query"]
        
        # Generate hypothetical document
        hypothetical_response = self.hypothetical_doc_prompt.invoke({
            "query": query,
            "condition": parsed_query.condition,
            "treatment_type": parsed_query.treatment_type,
            "patient_population": parsed_query.patient_population
        })
        print(hypothetical_response)
        hypothetical_doc = hypothetical_response.messages
        
        print(f"   📄 Generated hypothetical paper ({len(hypothetical_doc)} characters)")
        print(f"   🎯 Preview: {hypothetical_doc[:200]}...")
        
        state["hypothetical_document"] = hypothetical_doc
        
        return state
    
    def step3_embed_hypothetical_document(self, state: HyDEMedicalState) -> HyDEMedicalState:
        """
        STEP 3: EMBED HYPOTHETICAL DOCUMENT
        Convert the hypothetical document into embeddings that can be used for similarity search.
        This is what makes HyDE different from traditional RAG.
        """
        print("\n🧠 STEP 3: Creating embeddings for hypothetical document...")
        
        hypothetical_doc = state["hypothetical_document"]
        
        # Generate embeddings for the hypothetical document
        embeddings = self.embeddings.embed_query(hypothetical_doc)
        
        print(f"   🔢 Generated {len(embeddings)} dimensional embedding vector")
        print(f"   📊 Embedding sample: {embeddings[:5]}...")
        
        state["search_embeddings"] = embeddings
        
        return state
    
    def step4_retrieve_with_hypothetical_embeddings(self, state: HyDEMedicalState) -> HyDEMedicalState:
        """
        STEP 4: RETRIEVE USING HYPOTHETICAL EMBEDDINGS
        Use the hypothetical document embeddings to find similar real documents.
        This often finds more relevant results than searching with the original query.
        """
        print("\n🔍 STEP 4: Retrieving documents using hypothetical embeddings...")
        
        retrieved_docs = []
        
        # Retrieve from medical knowledge base using hypothetical document
        if self.medical_vectorstore:
            try:
                # Use the hypothetical document for retrieval instead of original query
                docs = self.medical_vectorstore.invoke(state["hypothetical_document"])
                retrieved_docs.extend([{"source": "knowledge_base", "content": doc} for doc in docs])
                print(f"   📚 Retrieved {len(docs)} documents from medical knowledge base")
            except Exception as e:
                print(f"   ⚠️ Error retrieving from knowledge base: {e}")
        
        # Also search for recent developments using web search
        try:
            # Use original query for web search to get current information
            web_results = self.web_search.invoke({"query": f"recent medical research {state['original_query']} 2024"})
            retrieved_docs.extend([{"source": "web", "content": result.get("content", "")} for result in web_results])
            print(f"   🌐 Retrieved {len(web_results)} recent research articles from web")
        except Exception as e:
            print(f"   ⚠️ Error in web search: {e}")
        
        print(f"   📊 Total retrieved documents: {len(retrieved_docs)}")
        
        state["retrieved_documents"] = retrieved_docs
        
        return state
    
    def step5_generate_final_response(self, state: HyDEMedicalState) -> HyDEMedicalState:
        """
        STEP 5: GENERATE FINAL RESPONSE
        Combine the retrieved documents with the original query to generate
        a comprehensive medical research response.
        """
        print("\n💬 STEP 5: Generating final medical research response...")
        
        query = state["original_query"]
        documents = state["retrieved_documents"]
        
        # Format documents for the prompt
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            source = doc.get("source", "unknown")
            content = doc.get("content", "")
            formatted_docs.append(f"Document {i} ({source}):\n{content}\n")
        
        documents_text = "\n".join(formatted_docs)
        
        # Generate comprehensive response
        response = self.response_generation_prompt.invoke({
            "query": query,
            "documents": documents_text
        })
        
        final_response = response.content
        
        print(f"   ✅ Generated comprehensive response ({len(final_response)} characters)")
        
        state["final_response"] = final_response
        
        return state
    
    def run_hyde_workflow(self, medical_query: str) -> Dict[str, Any]:
        """
        Execute the complete HyDE workflow for medical research.
        
        Args:
            medical_query: The medical research question
            
        Returns:
            Complete workflow results including hypothetical document and final response
        """
        print(f"\n🏥 STARTING HyDE MEDICAL RESEARCH WORKFLOW")
        print(f"Query: {medical_query}")
        print("=" * 80)
        
        # Initialize state
        state = HyDEMedicalState(
            original_query=medical_query,
            parsed_query=None,
            hypothetical_document="",
            retrieved_documents=[],
            final_response="",
            search_embeddings=[]
        )
        
        # Execute workflow steps
        state = self.step1_analyze_query(state)
        state = self.step2_generate_hypothetical_document(state)
        state = self.step3_embed_hypothetical_document(state)
        state = self.step4_retrieve_with_hypothetical_embeddings(state)
        state = self.step5_generate_final_response(state)
        
        print("\n✅ HyDE WORKFLOW COMPLETED")
        print("=" * 80)
        
        return state

# Example Usage and Demonstrations
def run_medical_research_examples():
    """
    Demonstrate HyDE RAG with various medical research scenarios.
    """
    
    # Initialize the HyDE system
    hyde_system = HyDEMedicalRAG()
    
    # Test cases representing different types of medical queries
    test_queries = [
        {
            "query": "What are the latest immunotherapy approaches for treating pediatric acute lymphoblastic leukemia with resistance to conventional chemotherapy?",
            "description": "Complex pediatric oncology query requiring recent research"
        },
        {
            "query": "How effective is CAR-T cell therapy compared to traditional chemotherapy for relapsed B-cell lymphomas in elderly patients?",
            "description": "Comparative effectiveness research question"
        },
        {
            "query": "What are the current clinical trials investigating combination immunotherapy for triple-negative breast cancer?",
            "description": "Current research and clinical trials query"
        }
    ]
    
    # Run each test case
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n{'#' * 20} EXAMPLE {i} {'#' * 20}")
        print(f"Description: {test_case['description']}")
        print(f"Query: {test_case['query']}")
        
        # Execute HyDE workflow
        result = hyde_system.run_hyde_workflow(test_case['query'])
        
        # Display results
        print(f"\n📋 WORKFLOW SUMMARY:")
        print(f"   Original Query: {result['original_query']}")
        print(f"   Parsed Condition: {result['parsed_query'].condition}")
        print(f"   Treatment Type: {result['parsed_query'].treatment_type}")
        print(f"   Documents Retrieved: {len(result['retrieved_documents'])}")
        
        print(f"\n📝 HYPOTHETICAL DOCUMENT PREVIEW:")
        print(result['hypothetical_document'][:500] + "...")
        
        print(f"\n🎯 FINAL RESPONSE:")
        print(result['final_response'])
        
        print(f"\n{'#' * 50}")

def compare_hyde_vs_traditional_rag():
    """
    Demonstrate the difference between HyDE and traditional RAG approaches.
    """
    
    print("\n🔬 COMPARISON: HyDE vs Traditional RAG")
    print("=" * 60)
    
    query = "What are breakthrough immunotherapy treatments for chemotherapy-resistant pediatric leukemia?"
    
    hyde_system = HyDEMedicalRAG()
    
    print(f"Query: {query}\n")
    
    # Traditional RAG approach (direct query search)
    print("🔍 TRADITIONAL RAG APPROACH:")
    print("   - Searches directly with user query")
    print("   - May miss relevant papers due to vocabulary mismatch")
    print("   - Limited by exact keyword matching")
    
    if hyde_system.medical_vectorstore:
        traditional_results = hyde_system.medical_vectorstore.invoke(query)
        print(f"   - Retrieved {len(traditional_results)} documents")
        print(f"   - Sample result: {str(traditional_results[0])[:200]}...")
    
    print("\n🧠 HyDE RAG APPROACH:")
    print("   - First generates hypothetical ideal research paper")
    print("   - Uses rich medical terminology and context")
    print("   - Retrieves based on semantic similarity to ideal answer")
    
    # Run HyDE workflow
    hyde_result = hyde_system.run_hyde_workflow(query)
    print(f"   - Generated hypothetical document ({len(hyde_result['hypothetical_document'])} chars)")
    print(f"   - Retrieved {len(hyde_result['retrieved_documents'])} documents")
    
    print("\n📊 KEY DIFFERENCES:")
    print("   1. HyDE generates domain-specific hypothetical content")
    print("   2. Better semantic matching through rich context")
    print("   3. More relevant results for complex medical queries")
    print("   4. Handles terminology variations better")

if __name__ == "__main__":
    """
    Main execution - demonstrates HyDE RAG for medical research.
    """
    
    print("🏥 HyDE RAG: Medical Research Assistant")
    print("=" * 50)
    print("This system demonstrates Hypothetical Document Embedding (HyDE)")
    print("for medical research queries. HyDE improves retrieval by first")
    print("generating hypothetical ideal research papers, then using them")
    print("to guide document retrieval.")
    print()
    
    # Run comprehensive examples
    run_medical_research_examples()
    
    # Show comparison with traditional RAG
    compare_hyde_vs_traditional_rag()
    
    print("\n🎯 CONCLUSION:")
    print("HyDE RAG is particularly powerful for medical research because:")
    print("1. Medical queries are complex and domain-specific")
    print("2. Hypothetical documents provide rich semantic context")
    print("3. Better retrieval of relevant research papers")
    print("4. Handles medical terminology and concepts effectively")
    print("5. Bridges the gap between natural language queries and technical literature")