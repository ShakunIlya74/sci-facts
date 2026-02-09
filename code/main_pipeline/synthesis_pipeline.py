"""
Synthesis Pipeline.

Direct LLM processing for paper synthesis tasks.

Two modes of operation:

1. MCP-NATIVE (Task Queue): The task_listener receives pre-formatted papers_context
   from the MCP server, retrieves the prompt via `synthesis_build_prompt` MCP prompt,
   and sends it directly to the LLM. This does NOT use SynthesisPipeline at all.

2. AGENT-INITIATED: When the agent pipeline wants to perform synthesis on papers
   it collected itself, it can use SynthesisPipeline.process_with_prompt() to
   pass a pre-built prompt to the LLM, or use the legacy process() method
   with locally-built prompts.

Supports synthesis goals:
- summary: Comprehensive synthesis of findings
- fact_extraction: Extract key facts and claims
- topic_extraction: Identify main topics and themes
- keywords_extraction: Extract relevant keywords

Supports synthesis tones:
- precise: Academic, thorough language
- fun_facts: Engaging, accessible presentation
- eli5: Simple, non-expert explanation
- brutal: Direct, critical analysis
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    """Result of synthesis processing."""
    
    success: bool
    content: Optional[str] = None
    error: Optional[str] = None
    paper_count: int = 0
    goal: str = ""
    tone: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "success": self.success,
            "content": self.content,
            "error": self.error,
            "paper_count": self.paper_count,
            "goal": self.goal,
            "tone": self.tone,
        }


class SynthesisPipeline:
    """
    Pipeline for processing synthesis tasks.
    
    Uses MCP server prompts and LLM inference to generate paper syntheses.
    """
    
    # Goal-specific system instructions
    GOAL_INSTRUCTIONS = {
        "summary": (
            "You are synthesizing academic papers to provide a comprehensive overview. "
            "Identify main findings, methodologies, and contributions. "
            "Highlight common themes, agreements, and disagreements across papers. "
            "Structure your response with clear sections."
        ),
        "fact_extraction": (
            "You are extracting key factual claims from academic papers. "
            "List specific findings as structured facts. "
            "Include paper reference numbers for each claim. "
            "Distinguish between well-established findings and emerging claims."
        ),
        "topic_extraction": (
            "You are identifying research topics and themes across papers. "
            "Create a hierarchical organization of topics. "
            "Identify main research areas and their subtopics. "
            "Note emerging or under-explored areas."
        ),
        "keywords_extraction": (
            "You are extracting key terms and concepts from academic papers. "
            "Group keywords by category (methods, applications, concepts). "
            "Rank by frequency and importance. "
            "Include both technical terms and broader concepts."
        ),
    }
    
    # Tone-specific modifiers
    TONE_MODIFIERS = {
        "precise": (
            "Use precise academic language. Be thorough and methodical. "
            "Cite specific papers when making claims using [N] notation. "
            "Maintain scholarly objectivity throughout."
        ),
        "fun_facts": (
            "Present information in an engaging, accessible way. "
            "Highlight surprising or counterintuitive findings. "
            "Use vivid examples while maintaining accuracy. "
            "Make the content interesting for a general educated audience."
        ),
        "eli5": (
            "Explain concepts as if to a curious but non-expert reader. "
            "Use simple language and everyday analogies. "
            "Avoid jargon; when technical terms are needed, explain them. "
            "Focus on the 'so what' - why these findings matter."
        ),
        "brutal": (
            "Be direct and critical in your analysis. "
            "Cut through unnecessary complexity - what actually matters? "
            "Point out limitations, gaps, and overstated claims. "
            "Don't hedge - give clear assessments."
        ),
    }
    
    def __init__(
        self,
        llm: ChatOpenAI,
        max_papers: int = 20,
        max_abstract_length: int = 500,
    ):
        """
        Initialize synthesis pipeline.
        
        Args:
            llm: LangChain ChatOpenAI instance configured for vLLM
            max_papers: Maximum papers to include in context
            max_abstract_length: Maximum characters per abstract
        """
        self.llm = llm
        self.max_papers = max_papers
        self.max_abstract_length = max_abstract_length
    
    def build_paper_context(self, papers: list[dict]) -> str:
        """
        Build concatenated paper context from paper list.
        
        Args:
            papers: List of paper dictionaries with title, authors, abstract
            
        Returns:
            Formatted string with all paper information
        """
        context_parts = []
        
        for i, paper in enumerate(papers[:self.max_papers], 1):
            title = paper.get("title", "Unknown Title")
            authors = paper.get("authors", "Unknown Authors")
            abstract = paper.get("abstract", "No abstract available")
            
            # Truncate abstract if too long
            if len(abstract) > self.max_abstract_length:
                abstract = abstract[:self.max_abstract_length] + "..."
            
            paper_text = f"[{i}] {title}\n"
            paper_text += f"Authors: {authors}\n"
            paper_text += f"Abstract: {abstract}\n"
            
            # Include optional metadata
            if paper.get("publication_date"):
                paper_text += f"Published: {paper['publication_date']}\n"
            if paper.get("citations"):
                paper_text += f"Citations: {paper['citations']}\n"
            
            context_parts.append(paper_text)
        
        return "\n---\n".join(context_parts)
    
    def build_prompt(self, papers: list[dict], goal: str, tone: str) -> tuple[str, str]:
        """
        Build system and user prompts for synthesis.
        
        Args:
            papers: List of paper dictionaries
            goal: Synthesis goal
            tone: Synthesis tone
            
        Returns:
            Tuple of (system_message, user_message)
        """
        # Get goal-specific instructions
        goal_instruction = self.GOAL_INSTRUCTIONS.get(
            goal, self.GOAL_INSTRUCTIONS["summary"]
        )
        
        # Get tone-specific modifier
        tone_modifier = self.TONE_MODIFIERS.get(
            tone, self.TONE_MODIFIERS["precise"]
        )
        
        # Build system message
        system_message = f"""You are an expert research synthesizer for academic literature.

{goal_instruction}

Writing style: {tone_modifier}

You will be given a collection of academic papers with their titles, authors, and abstracts. 
Analyze them according to the instructions above and provide your synthesis."""
        
        # Build user message with paper context
        paper_context = self.build_paper_context(papers)
        
        user_message = f"""Please synthesize the following {len(papers)} academic papers:

{paper_context}

---

Provide your {goal.replace('_', ' ')} synthesis following the guidelines above."""
        
        return system_message, user_message
    
    def process(
        self,
        papers: list[dict],
        goal: str,
        tone: str,
    ) -> SynthesisResult:
        """
        Process synthesis request.
        
        Args:
            papers: List of paper dictionaries with title, authors, abstract
            goal: Synthesis goal (summary, fact_extraction, topic_extraction, keywords_extraction)
            tone: Synthesis tone (precise, fun_facts, eli5, brutal)
            
        Returns:
            SynthesisResult with generated content or error
        """
        logger.info(f"Processing synthesis: goal={goal}, tone={tone}, papers={len(papers)}")
        
        # Validate inputs
        if not papers:
            return SynthesisResult(
                success=False,
                error="No papers provided for synthesis",
                goal=goal,
                tone=tone,
            )
        
        if goal not in self.GOAL_INSTRUCTIONS:
            return SynthesisResult(
                success=False,
                error=f"Invalid goal: {goal}. Valid: {list(self.GOAL_INSTRUCTIONS.keys())}",
                goal=goal,
                tone=tone,
            )
        
        if tone not in self.TONE_MODIFIERS:
            return SynthesisResult(
                success=False,
                error=f"Invalid tone: {tone}. Valid: {list(self.TONE_MODIFIERS.keys())}",
                goal=goal,
                tone=tone,
            )
        
        try:
            # Build prompts
            system_msg, user_msg = self.build_prompt(papers, goal, tone)
            
            # Invoke LLM
            messages = [
                SystemMessage(content=system_msg),
                HumanMessage(content=user_msg),
            ]
            
            logger.debug(f"Invoking LLM with {len(messages)} messages")
            response = self.llm.invoke(messages)
            
            content = response.content
            logger.info(f"LLM raw response type: {type(content)}, length: {len(content) if content else 0}")
            
            # Log truncated response for debugging
            if content:
                preview = content[:500] + "..." if len(content) > 500 else content
                logger.info(f"LLM response preview: {preview}")
            else:
                logger.warning("LLM returned empty/None content")
            
            if not content:
                return SynthesisResult(
                    success=False,
                    error="LLM returned empty response",
                    paper_count=len(papers),
                    goal=goal,
                    tone=tone,
                )
            
            logger.info(f"Synthesis completed: {len(content)} characters")
            
            return SynthesisResult(
                success=True,
                content=content,
                paper_count=len(papers),
                goal=goal,
                tone=tone,
            )
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return SynthesisResult(
                success=False,
                error=str(e),
                paper_count=len(papers),
                goal=goal,
                tone=tone,
            )
    
    async def process_async(
        self,
        papers: list[dict],
        goal: str,
        tone: str,
    ) -> SynthesisResult:
        """
        Async version of process for use with asyncio.
        
        Note: LangChain's ainvoke is used for async operation.
        """
        logger.info(f"Processing synthesis (async): goal={goal}, tone={tone}, papers={len(papers)}")
        
        # Validate inputs
        if not papers:
            return SynthesisResult(
                success=False,
                error="No papers provided for synthesis",
                goal=goal,
                tone=tone,
            )
        
        if goal not in self.GOAL_INSTRUCTIONS:
            return SynthesisResult(
                success=False,
                error=f"Invalid goal: {goal}. Valid: {list(self.GOAL_INSTRUCTIONS.keys())}",
                goal=goal,
                tone=tone,
            )
        
        if tone not in self.TONE_MODIFIERS:
            return SynthesisResult(
                success=False,
                error=f"Invalid tone: {tone}. Valid: {list(self.TONE_MODIFIERS.keys())}",
                goal=goal,
                tone=tone,
            )
        
        try:
            # Build prompts
            system_msg, user_msg = self.build_prompt(papers, goal, tone)
            
            # Invoke LLM asynchronously
            messages = [
                SystemMessage(content=system_msg),
                HumanMessage(content=user_msg),
            ]
            
            logger.debug(f"Invoking LLM async with {len(messages)} messages")
            response = await self.llm.ainvoke(messages)
            
            content = response.content
            logger.info(f"LLM raw response type: {type(content)}, length: {len(content) if content else 0}")
            
            # Log truncated response for debugging
            if content:
                preview = content[:500] + "..." if len(content) > 500 else content
                logger.info(f"LLM response preview: {preview}")
            else:
                logger.warning("LLM returned empty/None content")
            
            if not content:
                return SynthesisResult(
                    success=False,
                    error="LLM returned empty response",
                    paper_count=len(papers),
                    goal=goal,
                    tone=tone,
                )
            
            logger.info(f"Synthesis completed (async): {len(content)} characters")
            
            return SynthesisResult(
                success=True,
                content=content,
                paper_count=len(papers),
                goal=goal,
                tone=tone,
            )
            
        except Exception as e:
            logger.error(f"Synthesis failed (async): {e}")
            return SynthesisResult(
                success=False,
                error=str(e),
                paper_count=len(papers),
                goal=goal,
                tone=tone,
            )

    async def process_with_prompt(
        self,
        prompt_text: str,
        goal: str = "",
        tone: str = "",
        paper_count: int = 0,
    ) -> SynthesisResult:
        """
        Process synthesis using a pre-built prompt from the MCP server.

        This is used by the agent workflow when the prompt has already been
        retrieved via the `synthesis_build_prompt` MCP prompt.

        Args:
            prompt_text: Complete LLM-ready prompt from MCP server
            goal: Synthesis goal (for metadata only)
            tone: Synthesis tone (for metadata only)
            paper_count: Number of papers (for metadata only)

        Returns:
            SynthesisResult with generated content or error
        """
        logger.info(f"Processing synthesis with pre-built prompt: goal={goal}, tone={tone}")

        if not prompt_text:
            return SynthesisResult(
                success=False,
                error="Empty prompt text provided",
                goal=goal,
                tone=tone,
            )

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt_text)])
            content = response.content

            if not content:
                return SynthesisResult(
                    success=False,
                    error="LLM returned empty response",
                    paper_count=paper_count,
                    goal=goal,
                    tone=tone,
                )

            logger.info(f"Synthesis completed (pre-built prompt): {len(content)} characters")
            logger.debug(f"Synthesis content preview: {content[:500]}...")

            return SynthesisResult(
                success=True,
                content=content,
                paper_count=paper_count,
                goal=goal,
                tone=tone,
            )

        except Exception as e:
            logger.error(f"Synthesis failed (pre-built prompt): {e}")
            return SynthesisResult(
                success=False,
                error=str(e),
                paper_count=paper_count,
                goal=goal,
                tone=tone,
            )


def create_synthesis_pipeline(
    host: str = "127.0.0.1",
    port: int = 8000,
    api_key: str = "token-abc123",
    model: str = "Qwen/Qwen3-VL-4B-Instruct",
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> SynthesisPipeline:
    """
    Create a synthesis pipeline with configured LLM.
    
    Args:
        host: vLLM server host
        port: vLLM server port
        api_key: vLLM server API key
        model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        
    Returns:
        Configured SynthesisPipeline instance
    """
    llm = ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base=f"http://{host}:{port}/v1",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    return SynthesisPipeline(llm=llm)
