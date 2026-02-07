"""
Agent Chat Pipeline for Literature Review.

A linear multi-step pipeline using LangGraph for conducting AI-assisted literature reviews.

Pipeline Steps:
1. Keyword Extraction: Extract search keywords from user query
2. Keyword Search: Execute search and select relevant papers
3. Query Augmentation: Expand and refine the search query
4. Semantic Search: Find semantically similar papers
5. Advanced Search: Generate custom search query based on findings

The pipeline sends intermediate results to the MCP server during processing
and marks the session as complete when done.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


# =============================================================================
# State Definitions
# =============================================================================

class AgentState(TypedDict):
    """State maintained throughout the agent pipeline."""
    
    # Input
    user_query: str
    task_id: str
    
    # Extracted information
    keywords: list[str]
    augmented_query: str
    custom_search_query: str
    
    # Paper collections
    keyword_search_results: list[dict]  # Raw search results
    keyword_selected_papers: list[dict]  # Agent-selected papers from keyword search
    semantic_search_results: list[dict]  # Raw semantic search results
    semantic_selected_papers: list[dict]  # Agent-selected from semantic search
    advanced_search_results: list[dict]  # Results from custom query search
    
    # All collected papers (accumulated)
    all_selected_papers: list[dict]
    all_paper_ids: list[int]
    
    # Status
    current_step: str
    error: Optional[str]
    completed: bool


@dataclass
class AgentResult:
    """Final result of agent pipeline execution."""
    
    success: bool
    task_id: str
    user_query: str
    all_selected_papers: list[dict] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    augmented_query: str = ""
    custom_search_query: str = ""
    error: Optional[str] = None
    steps_completed: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "success": self.success,
            "task_id": self.task_id,
            "user_query": self.user_query,
            "all_selected_papers": self.all_selected_papers,
            "paper_count": len(self.all_selected_papers),
            "keywords": self.keywords,
            "augmented_query": self.augmented_query,
            "custom_search_query": self.custom_search_query,
            "error": self.error,
            "steps_completed": self.steps_completed,
        }


# =============================================================================
# Prompts
# =============================================================================

KEYWORD_EXTRACTION_PROMPT = """You are a research assistant specializing in academic literature search.

Given a user's research question or topic, extract the most effective search keywords.

Guidelines:
- Extract 3-7 distinct keywords or short phrases
- Include both specific technical terms and broader concepts
- Consider synonyms and alternative terminology
- Prioritize terms likely to appear in paper titles and abstracts
- Remove stop words and overly generic terms

Output format:
Return ONLY a JSON array of keywords, nothing else.
Example: ["machine learning", "neural networks", "deep learning", "classification"]

User query: {user_query}

Keywords:"""

PAPER_SELECTION_PROMPT = """You are a research assistant evaluating papers for a literature review.

User's research query: {user_query}

Below are {n_papers} papers from a keyword search. Select the most relevant papers for this research query.

Papers:
{papers_list}

Guidelines:
- Select papers that directly address the research query
- Consider methodology, findings, and applicability
- Prefer recent, high-impact papers when relevance is similar
- Aim to select 3-5 papers, but can select fewer if few are truly relevant

Output format:
Return ONLY a JSON array of paper IDs (numbers) that you select, nothing else.
Example: [12345, 67890, 11111]

Selected paper IDs:"""

QUERY_AUGMENTATION_PROMPT = """You are a research assistant improving search queries for academic literature.

Original user query: {user_query}
Keywords extracted: {keywords}

Based on the initial search and these selected papers:
{selected_papers}

Augment the query to:
1. Include related concepts discovered from the papers
2. Add methodological terms that might find similar work
3. Expand scope to catch related but differently-worded research
4. Maintain focus on the original research intent

Output format:
Return a single augmented search query (1-3 sentences) that can be used for semantic search.
Do not include any explanation, just the query.

Augmented query:"""

SEMANTIC_SELECTION_PROMPT = """You are a research assistant evaluating papers for a literature review.

User's research query: {user_query}
Augmented search query: {augmented_query}

Below are {n_papers} papers from semantic search. Select the most relevant papers.

Papers already selected from keyword search:
{already_selected}

New papers from semantic search:
{papers_list}

Guidelines:
- Select papers that complement the already selected papers
- Look for different perspectives, methods, or applications
- Avoid papers too similar to what's already selected
- Select 2-5 papers that add value to the collection

Output format:
Return ONLY a JSON array of paper IDs (numbers) that you select, nothing else.
Example: [12345, 67890]

Selected paper IDs:"""

CUSTOM_SEARCH_PROMPT = """You are a research strategist designing targeted literature searches.

User's original query: {user_query}
Papers already collected:
{collected_papers}

Based on the collected papers and the research goal, design a new search query that will find:
- Papers that the previous searches might have missed
- Important related work from different subfields
- Seminal papers that influenced this area

Think about gaps in the current collection and how to fill them.

Output format:
Return a single search query (1-2 sentences) for semantic search.
Do not include any explanation, just the query.

Custom search query:"""


# =============================================================================
# Agent Chat Pipeline
# =============================================================================

class AgentChatPipeline:
    """
    Multi-step literature review pipeline.
    
    Uses LLM to guide paper discovery through keyword and semantic search,
    with intelligent paper selection at each step.
    """
    
    def __init__(
        self,
        llm: ChatOpenAI,
        mcp_client: Any,  # MCPClient instance
        top_k_per_search: int = 5,
    ):
        """
        Initialize agent pipeline.
        
        Args:
            llm: LangChain ChatOpenAI instance
            mcp_client: MCPClient instance for tool calls
            top_k_per_search: Number of papers to retrieve per search
        """
        self.llm = llm
        self.mcp_client = mcp_client
        self.top_k = top_k_per_search
        self._graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("extract_keywords", self._extract_keywords)
        graph.add_node("keyword_search", self._keyword_search)
        graph.add_node("select_keyword_papers", self._select_keyword_papers)
        graph.add_node("augment_query", self._augment_query)
        graph.add_node("semantic_search", self._semantic_search)
        graph.add_node("select_semantic_papers", self._select_semantic_papers)
        graph.add_node("advanced_search", self._advanced_search)
        graph.add_node("finalize", self._finalize)
        
        # Define edges (linear flow)
        graph.set_entry_point("extract_keywords")
        graph.add_edge("extract_keywords", "keyword_search")
        graph.add_edge("keyword_search", "select_keyword_papers")
        graph.add_edge("select_keyword_papers", "augment_query")
        graph.add_edge("augment_query", "semantic_search")
        graph.add_edge("semantic_search", "select_semantic_papers")
        graph.add_edge("select_semantic_papers", "advanced_search")
        graph.add_edge("advanced_search", "finalize")
        graph.add_edge("finalize", END)
        
        return graph.compile()
    
    async def _extract_keywords(self, state: AgentState) -> AgentState:
        """Extract search keywords from user query."""
        logger.info(f"Step: extract_keywords for task {state['task_id']}")
        state["current_step"] = "extract_keywords"
        
        try:
            prompt = KEYWORD_EXTRACTION_PROMPT.format(user_query=state["user_query"])
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            
            # Parse keywords from response
            content = response.content.strip()
            logger.info(f"LLM response for keyword extraction: {content[:300]}..." if len(content) > 300 else f"LLM response for keyword extraction: {content}")
            
            # Try to extract JSON array from response
            if content.startswith("["):
                keywords = json.loads(content)
            else:
                # Fallback: split by common delimiters
                keywords = [k.strip().strip('"\'') for k in content.split(",")]
            
            state["keywords"] = keywords
            logger.info(f"Extracted keywords: {keywords}")
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            state["keywords"] = state["user_query"].split()[:5]  # Fallback
            state["error"] = f"Keyword extraction error: {e}"
        
        return state
    
    async def _keyword_search(self, state: AgentState) -> AgentState:
        """Execute keyword search using MCP tools."""
        logger.info(f"Step: keyword_search for task {state['task_id']}")
        state["current_step"] = "keyword_search"
        
        try:
            # Combine keywords into search query
            query = " ".join(state["keywords"][:5])  # Use top 5 keywords
            
            result = await self.mcp_client.papers_search_keyword(
                query=query,
                n_results=self.top_k,
            )
            
            papers = result.get("papers", [])
            state["keyword_search_results"] = papers
            logger.info(f"Keyword search found {len(papers)} papers")
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            state["keyword_search_results"] = []
            state["error"] = f"Keyword search error: {e}"
        
        return state
    
    async def _select_keyword_papers(self, state: AgentState) -> AgentState:
        """LLM selects relevant papers from keyword search results."""
        logger.info(f"Step: select_keyword_papers for task {state['task_id']}")
        state["current_step"] = "select_keyword_papers"
        
        papers = state["keyword_search_results"]
        if not papers:
            state["keyword_selected_papers"] = []
            return state
        
        try:
            # Format papers for LLM
            papers_list = self._format_papers_for_selection(papers)
            
            prompt = PAPER_SELECTION_PROMPT.format(
                user_query=state["user_query"],
                n_papers=len(papers),
                papers_list=papers_list,
            )
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            logger.info(f"LLM response for paper selection: {content[:300]}..." if len(content) > 300 else f"LLM response for paper selection: {content}")
            
            # Parse selected paper IDs
            selected_ids = self._parse_paper_ids(content)
            logger.debug(f"Parsed paper IDs: {selected_ids}")
            
            # Get selected papers
            selected = [p for p in papers if p.get("paper_id") in selected_ids]
            state["keyword_selected_papers"] = selected
            
            # Add to accumulated papers
            state["all_selected_papers"].extend(selected)
            state["all_paper_ids"].extend([p.get("paper_id") for p in selected])
            
            logger.info(f"Selected {len(selected)} papers from keyword search")
            
            # Send intermediate results to MCP server
            await self._send_intermediate_results(state, "keyword_search")
            
        except Exception as e:
            logger.error(f"Paper selection failed: {e}")
            # Select top papers as fallback
            state["keyword_selected_papers"] = papers[:3]
            state["all_selected_papers"].extend(papers[:3])
            state["all_paper_ids"].extend([p.get("paper_id") for p in papers[:3]])
        
        return state
    
    async def _augment_query(self, state: AgentState) -> AgentState:
        """Augment the search query based on selected papers."""
        logger.info(f"Step: augment_query for task {state['task_id']}")
        state["current_step"] = "augment_query"
        
        try:
            # Format selected papers
            selected_papers = "\n".join([
                f"- {p.get('title', 'Unknown')}" 
                for p in state["keyword_selected_papers"][:5]
            ])
            
            prompt = QUERY_AUGMENTATION_PROMPT.format(
                user_query=state["user_query"],
                keywords=", ".join(state["keywords"]),
                selected_papers=selected_papers,
            )
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            augmented = response.content.strip()
            logger.info(f"LLM response for query augmentation: {augmented[:300]}..." if len(augmented) > 300 else f"LLM response for query augmentation: {augmented}")
            state["augmented_query"] = augmented
            logger.info(f"Augmented query: {state['augmented_query'][:100]}...")
            
        except Exception as e:
            logger.error(f"Query augmentation failed: {e}")
            state["augmented_query"] = state["user_query"]
        
        return state
    
    async def _semantic_search(self, state: AgentState) -> AgentState:
        """Execute semantic search with augmented query."""
        logger.info(f"Step: semantic_search for task {state['task_id']}")
        state["current_step"] = "semantic_search"
        
        try:
            result = await self.mcp_client.papers_search_by_text(
                query=state["augmented_query"],
                n_results=self.top_k,
            )
            
            papers = result.get("papers", [])
            
            # Filter out already selected papers
            existing_ids = set(state["all_paper_ids"])
            new_papers = [p for p in papers if p.get("paper_id") not in existing_ids]
            
            state["semantic_search_results"] = new_papers
            logger.info(f"Semantic search found {len(new_papers)} new papers")
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            state["semantic_search_results"] = []
        
        return state
    
    async def _select_semantic_papers(self, state: AgentState) -> AgentState:
        """LLM selects relevant papers from semantic search results."""
        logger.info(f"Step: select_semantic_papers for task {state['task_id']}")
        state["current_step"] = "select_semantic_papers"
        
        papers = state["semantic_search_results"]
        if not papers:
            state["semantic_selected_papers"] = []
            return state
        
        try:
            # Format papers
            papers_list = self._format_papers_for_selection(papers)
            already_selected = "\n".join([
                f"- {p.get('title', 'Unknown')}" 
                for p in state["keyword_selected_papers"]
            ])
            
            prompt = SEMANTIC_SELECTION_PROMPT.format(
                user_query=state["user_query"],
                augmented_query=state["augmented_query"],
                n_papers=len(papers),
                already_selected=already_selected or "None yet",
                papers_list=papers_list,
            )
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            logger.info(f"LLM response for semantic paper selection: {content[:300]}..." if len(content) > 300 else f"LLM response for semantic paper selection: {content}")
            
            # Parse selected paper IDs
            selected_ids = self._parse_paper_ids(content)
            logger.debug(f"Parsed semantic paper IDs: {selected_ids}")
            
            # Get selected papers
            selected = [p for p in papers if p.get("paper_id") in selected_ids]
            state["semantic_selected_papers"] = selected
            
            # Add to accumulated papers
            state["all_selected_papers"].extend(selected)
            state["all_paper_ids"].extend([p.get("paper_id") for p in selected])
            
            logger.info(f"Selected {len(selected)} papers from semantic search")
            
            # Send intermediate results
            await self._send_intermediate_results(state, "semantic_search")
            
        except Exception as e:
            logger.error(f"Semantic paper selection failed: {e}")
            state["semantic_selected_papers"] = papers[:2]
            state["all_selected_papers"].extend(papers[:2])
            state["all_paper_ids"].extend([p.get("paper_id") for p in papers[:2]])
        
        return state
    
    async def _advanced_search(self, state: AgentState) -> AgentState:
        """Generate custom search query and execute final search."""
        logger.info(f"Step: advanced_search for task {state['task_id']}")
        state["current_step"] = "advanced_search"
        
        try:
            # Format collected papers
            collected_papers = "\n".join([
                f"- {p.get('title', 'Unknown')}" 
                for p in state["all_selected_papers"][:10]
            ])
            
            prompt = CUSTOM_SEARCH_PROMPT.format(
                user_query=state["user_query"],
                collected_papers=collected_papers or "None yet",
            )
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            custom_query = response.content.strip()
            logger.info(f"LLM response for custom search: {custom_query[:300]}..." if len(custom_query) > 300 else f"LLM response for custom search: {custom_query}")
            state["custom_search_query"] = custom_query
            
            logger.info(f"Custom search query: {custom_query[:100]}...")
            
            # Execute search
            result = await self.mcp_client.papers_search_by_text(
                query=custom_query,
                n_results=self.top_k,
            )
            
            papers = result.get("papers", [])
            
            # Filter and select top papers not already selected
            existing_ids = set(state["all_paper_ids"])
            new_papers = [p for p in papers if p.get("paper_id") not in existing_ids][:3]
            
            state["advanced_search_results"] = new_papers
            state["all_selected_papers"].extend(new_papers)
            state["all_paper_ids"].extend([p.get("paper_id") for p in new_papers])
            
            logger.info(f"Advanced search added {len(new_papers)} papers")
            
            # Send intermediate results
            await self._send_intermediate_results(state, "advanced_search")
            
        except Exception as e:
            logger.error(f"Advanced search failed: {e}")
            state["advanced_search_results"] = []
        
        return state
    
    async def _finalize(self, state: AgentState) -> AgentState:
        """Finalize the pipeline and mark as complete."""
        logger.info(f"Step: finalize for task {state['task_id']}")
        state["current_step"] = "finalize"
        state["completed"] = True
        
        logger.info(
            f"Pipeline complete: {len(state['all_selected_papers'])} papers collected"
        )
        
        return state
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _format_papers_for_selection(self, papers: list[dict]) -> str:
        """Format papers for LLM selection prompt."""
        lines = []
        for p in papers:
            paper_id = p.get("paper_id", "?")
            title = p.get("title", "Unknown")
            authors = p.get("authors", "Unknown")[:100]  # Truncate authors
            abstract = p.get("abstract", "")[:300] if p.get("abstract") else "No abstract"
            citations = p.get("citations", 0)
            
            lines.append(
                f"ID: {paper_id}\n"
                f"Title: {title}\n"
                f"Authors: {authors}\n"
                f"Citations: {citations}\n"
                f"Abstract: {abstract}...\n"
            )
        return "\n---\n".join(lines)
    
    def _parse_paper_ids(self, content: str) -> list[int]:
        """Parse paper IDs from LLM response."""
        try:
            # Try to parse as JSON array
            if "[" in content:
                start = content.find("[")
                end = content.rfind("]") + 1
                ids = json.loads(content[start:end])
                return [int(i) for i in ids]
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback: extract numbers from content
        import re
        numbers = re.findall(r'\d+', content)
        return [int(n) for n in numbers if len(n) > 3]  # Assume paper IDs are > 3 digits
    
    async def _send_intermediate_results(self, state: AgentState, step: str):
        """Send intermediate results to MCP server."""
        try:
            # Use task_update_status to send progress
            paper_ids = state["all_paper_ids"]
            await self.mcp_client.call_tool(
                "task_update_status",
                {
                    "task_id": state["task_id"],
                    "status": "processing",
                    "result": {
                        "step": step,
                        "papers_collected": len(paper_ids),
                        "paper_ids": paper_ids,
                    }
                }
            )
            logger.debug(f"Sent intermediate results for step {step}")
        except Exception as e:
            logger.warning(f"Failed to send intermediate results: {e}")
    
    # =========================================================================
    # Public Interface
    # =========================================================================
    
    async def run(self, user_query: str, task_id: str) -> AgentResult:
        """
        Run the agent pipeline.
        
        Args:
            user_query: User's research question
            task_id: Task ID for tracking
            
        Returns:
            AgentResult with collected papers and metadata
        """
        logger.info(f"Starting agent pipeline for task {task_id}")
        logger.info(f"User query: {user_query}")
        
        # Initialize state
        initial_state: AgentState = {
            "user_query": user_query,
            "task_id": task_id,
            "keywords": [],
            "augmented_query": "",
            "custom_search_query": "",
            "keyword_search_results": [],
            "keyword_selected_papers": [],
            "semantic_search_results": [],
            "semantic_selected_papers": [],
            "advanced_search_results": [],
            "all_selected_papers": [],
            "all_paper_ids": [],
            "current_step": "",
            "error": None,
            "completed": False,
        }
        
        try:
            # Run the graph
            final_state = await self._graph.ainvoke(initial_state)
            
            return AgentResult(
                success=True,
                task_id=task_id,
                user_query=user_query,
                all_selected_papers=final_state.get("all_selected_papers", []),
                keywords=final_state.get("keywords", []),
                augmented_query=final_state.get("augmented_query", ""),
                custom_search_query=final_state.get("custom_search_query", ""),
                steps_completed=[
                    "extract_keywords",
                    "keyword_search", 
                    "select_keyword_papers",
                    "augment_query",
                    "semantic_search",
                    "select_semantic_papers",
                    "advanced_search",
                    "finalize",
                ],
            )
            
        except Exception as e:
            logger.error(f"Agent pipeline failed: {e}")
            return AgentResult(
                success=False,
                task_id=task_id,
                user_query=user_query,
                error=str(e),
            )


def create_agent_pipeline(
    mcp_client: Any,
    host: str = "127.0.0.1",
    port: int = 8000,
    api_key: str = "token-abc123",
    model: str = "Qwen/Qwen3-VL-4B-Instruct",
    temperature: float = 0.2,
    max_tokens: int = 1024,
    top_k_per_search: int = 5,
) -> AgentChatPipeline:
    """
    Create an agent pipeline with configured LLM and MCP client.
    
    Args:
        mcp_client: MCPClient instance for tool calls
        host: vLLM server host
        port: vLLM server port
        api_key: vLLM server API key
        model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        top_k_per_search: Papers to retrieve per search
        
    Returns:
        Configured AgentChatPipeline instance
    """
    llm = ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base=f"http://{host}:{port}/v1",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    return AgentChatPipeline(
        llm=llm,
        mcp_client=mcp_client,
        top_k_per_search=top_k_per_search,
    )
