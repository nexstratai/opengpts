from enum import Enum
from functools import lru_cache
import os
from typing import Annotated, Dict, List, Literal, Optional

from langchain.tools.retriever import create_retriever_tool
from langchain.tools import StructuredTool
from langchain_community.agent_toolkits.connery import ConneryToolkit
from langchain_community.retrievers.kay import KayAiRetriever
from langchain_community.retrievers.pubmed import PubMedRetriever
from langchain_community.retrievers.wikipedia import WikipediaRetriever
from langchain_community.retrievers.you import YouRetriever
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.connery import ConneryService
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import (
    TavilyAnswer as _TavilyAnswer,
)
from langchain_community.tools.tavily_search import (
    TavilySearchResults,
)
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_azure_dynamic_sessions import SessionsPythonREPLTool

from langchain_core.tools import Tool
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from app.upload import vstore


class DDGInput(BaseModel):
    query: Annotated[str, Field(description="search query to look up")]


class ArxivInput(BaseModel):
    query: Annotated[str, Field(description="search query to look up")]


class PythonREPLInput(BaseModel):
    query: Annotated[str, Field(description="python command to run")]


class DallEInput(BaseModel):
    query: Annotated[str, Field(description="image description to generate image from")]


class AvailableTools(str, Enum):
    ACTION_SERVER = "action_server_by_sema4ai"
    CONNERY = "ai_action_runner_by_connery"
    DDG_SEARCH = "ddg_search"
    TAVILY = "search_tavily"
    TAVILY_ANSWER = "search_tavily_answer"
    RETRIEVAL = "retrieval"
    ARXIV = "arxiv"
    YOU_SEARCH = "you_search"
    SEC_FILINGS = "sec_filings_kai_ai"
    PRESS_RELEASES = "press_releases_kai_ai"
    PUBMED = "pubmed"
    WIKIPEDIA = "wikipedia"
    DALL_E = "dall_e"
    ISSUE_TREE = "issue_tree"
    STRATEGIC_APPROACH = "strategic_approach"
    DECISION_TREE = "decision_tree"
    RISKS_REWARDS = "risks_rewards"
    COST_BENEFITS = "cost_benefits"
    PROS_CONS = "pros_cons"
    ROOT_CAUSE = "root_cause"
    SWOT = "swot"
    MULTI_CRITERIA = "multi_criteria"
    TWO_BY_TWO = "two_by_two"
    MORPH_BOX = "morph_box"
    MATRIX = "matrix"
    TABLE_ANALYSIS = "table_analysis"
    PYTHON_REPL = "python_repl"
    AZURE_DYNAMIC_SESSIONS = "azure_dynamic_sessions"


class ToolConfig(TypedDict):
    ...


class BaseTool(BaseModel):
    type: AvailableTools
    name: str
    description: str
    config: ToolConfig = Field(default_factory=dict)
    multi_use: bool = False


class ActionServerConfig(ToolConfig):
    url: str
    api_key: str


class ActionServer(BaseTool):
    type: Literal[AvailableTools.ACTION_SERVER] = AvailableTools.ACTION_SERVER
    name: Literal["Action Server by Sema4.ai"] = "Action Server by Sema4.ai"
    description: Literal[
        (
            "Run AI actions with "
            "[Sema4.ai Action Server](https://github.com/Sema4AI/actions)."
        )
    ] = (
        "Run AI actions with "
        "[Sema4.ai Action Server](https://github.com/Sema4AI/actions)."
    )
    config: ActionServerConfig
    multi_use: Literal[True] = True


class Connery(BaseTool):
    type: Literal[AvailableTools.CONNERY] = AvailableTools.CONNERY
    name: Literal["AI Action Runner by Connery"] = "AI Action Runner by Connery"
    description: Literal[
        (
            "Connect OpenGPTs to the real world with "
            "[Connery](https://github.com/connery-io/connery)."
        )
    ] = (
        "Connect OpenGPTs to the real world with "
        "[Connery](https://github.com/connery-io/connery)."
    )


class DDGSearch(BaseTool):
    type: Literal[AvailableTools.DDG_SEARCH] = AvailableTools.DDG_SEARCH
    name: Literal["DuckDuckGo Search"] = "DuckDuckGo Search"
    description: Literal[
        "Search the web with [DuckDuckGo](https://pypi.org/project/duckduckgo-search/)."
    ] = "Search the web with [DuckDuckGo](https://pypi.org/project/duckduckgo-search/)."


class Arxiv(BaseTool):
    type: Literal[AvailableTools.ARXIV] = AvailableTools.ARXIV
    name: Literal["Arxiv"] = "Arxiv"
    description: Literal[
        "Searches [Arxiv](https://arxiv.org/)."
    ] = "Searches [Arxiv](https://arxiv.org/)."


class YouSearch(BaseTool):
    type: Literal[AvailableTools.YOU_SEARCH] = AvailableTools.YOU_SEARCH
    name: Literal["You.com Search"] = "You.com Search"
    description: Literal[
        "Uses [You.com](https://you.com/) search, optimized responses for LLMs."
    ] = "Uses [You.com](https://you.com/) search, optimized responses for LLMs."


class SecFilings(BaseTool):
    type: Literal[AvailableTools.SEC_FILINGS] = AvailableTools.SEC_FILINGS
    name: Literal["SEC Filings (Kay.ai)"] = "SEC Filings (Kay.ai)"
    description: Literal[
        "Searches through SEC filings using [Kay.ai](https://www.kay.ai/)."
    ] = "Searches through SEC filings using [Kay.ai](https://www.kay.ai/)."


class PressReleases(BaseTool):
    type: Literal[AvailableTools.PRESS_RELEASES] = AvailableTools.PRESS_RELEASES
    name: Literal["Press Releases (Kay.ai)"] = "Press Releases (Kay.ai)"
    description: Literal[
        "Searches through press releases using [Kay.ai](https://www.kay.ai/)."
    ] = "Searches through press releases using [Kay.ai](https://www.kay.ai/)."


class PubMed(BaseTool):
    type: Literal[AvailableTools.PUBMED] = AvailableTools.PUBMED
    name: Literal["PubMed"] = "PubMed"
    description: Literal[
        "Searches [PubMed](https://pubmed.ncbi.nlm.nih.gov/)."
    ] = "Searches [PubMed](https://pubmed.ncbi.nlm.nih.gov/)."


class Wikipedia(BaseTool):
    type: Literal[AvailableTools.WIKIPEDIA] = AvailableTools.WIKIPEDIA
    name: Literal["Wikipedia"] = "Wikipedia"
    description: Literal[
        "Searches [Wikipedia](https://pypi.org/project/wikipedia/)."
    ] = "Searches [Wikipedia](https://pypi.org/project/wikipedia/)."


class Tavily(BaseTool):
    type: Literal[AvailableTools.TAVILY] = AvailableTools.TAVILY
    name: Literal["Search (Tavily)"] = "Search (Tavily)"
    description: Literal[
        (
            "Uses the [Tavily](https://app.tavily.com/) search engine. "
            "Includes sources in the response."
        )
    ] = (
        "Uses the [Tavily](https://app.tavily.com/) search engine. "
        "Includes sources in the response."
    )


class TavilyAnswer(BaseTool):
    type: Literal[AvailableTools.TAVILY_ANSWER] = AvailableTools.TAVILY_ANSWER
    name: Literal["Search (short answer, Tavily)"] = "Search (short answer, Tavily)"
    description: Literal[
        (
            "Uses the [Tavily](https://app.tavily.com/) search engine. "
            "This returns only the answer, no supporting evidence."
        )
    ] = (
        "Uses the [Tavily](https://app.tavily.com/) search engine. "
        "This returns only the answer, no supporting evidence."
    )


class Retrieval(BaseTool):
    type: Literal[AvailableTools.RETRIEVAL] = AvailableTools.RETRIEVAL
    name: Literal["Retrieval"] = "Retrieval"
    description: Literal[
        "Look up information in uploaded files."
    ] = "Look up information in uploaded files."


class DallE(BaseTool):
    type: Literal[AvailableTools.DALL_E] = AvailableTools.DALL_E
    name: Literal["Generate Image (Dall-E)"] = "Generate Image (Dall-E)"
    description: Literal[
        "Generates images from a text description using OpenAI's DALL-E model."
    ] = "Generates images from a text description using OpenAI's DALL-E model."




class IssueNode(BaseModel):
    nodeLabel: str = Field(
        description="The label or description for this node of the issue tree."
    )
    children: Optional[List['IssueNode']] = Field(
        default_factory=list,
        description="An array of child nodes representing sub-issues or further breakdowns of the issue."
    )


IssueNode.update_forward_refs()


class IssueTreeInput(BaseModel):
    type: str = Field(
        description="Identifies the type of the structure as an issue tree.",
        pattern="^ISSUE_TREE$"
    )
    name: str = Field(
        description="Unique title for the assessment with 3-4 words, different than previous assessments"
    )
    data: IssueNode = Field(
        description="The root node of the issue tree, detailing the main issue and its hierarchical breakdown."
    )


RETRIEVAL_DESCRIPTION = """Can be used to look up information that was uploaded to this assistant.
If the user is referencing particular files, that is often a good hint that information may be here.
If the user asks a vague question, they are likely meaning to look up info from this retriever, and you should call it!"""


def get_retriever(assistant_id: str, thread_id: str):
    return vstore.as_retriever(
        search_kwargs={"filter": {"namespace": {"$in": [assistant_id, thread_id]}}}
    )


@lru_cache(maxsize=5)
def get_retrieval_tool(assistant_id: str, thread_id: str, description: str):
    return create_retriever_tool(
        get_retriever(assistant_id, thread_id),
        "Retriever",
        description,
    )


@lru_cache(maxsize=1)
def _get_duck_duck_go():
    return DuckDuckGoSearchRun(args_schema=DDGInput)


@lru_cache(maxsize=1)
def _get_arxiv():
    return ArxivQueryRun(api_wrapper=ArxivAPIWrapper(), args_schema=ArxivInput)


@lru_cache(maxsize=1)
def _get_you_search():
    return create_retriever_tool(
        YouRetriever(n_hits=3, n_snippets_per_hit=3),
        "you_search",
        "Searches for documents using You.com",
    )


@lru_cache(maxsize=1)
def _get_sec_filings():
    return create_retriever_tool(
        KayAiRetriever.create(
            dataset_id="company", data_types=["10-K", "10-Q"], num_contexts=3
        ),
        "sec_filings_search",
        "Search for a query among SEC Filings",
    )


@lru_cache(maxsize=1)
def _get_press_releases():
    return create_retriever_tool(
        KayAiRetriever.create(
            dataset_id="company", data_types=["PressRelease"], num_contexts=6
        ),
        "press_release_search",
        "Search for a query among press releases from US companies",
    )


@lru_cache(maxsize=1)
def _get_pubmed():
    return create_retriever_tool(
        PubMedRetriever(), "pub_med_search", "Search for a query on PubMed"
    )


@lru_cache(maxsize=1)
def _get_wikipedia():
    return create_retriever_tool(
        WikipediaRetriever(), "wikipedia", "Search for a query on Wikipedia"
    )


@lru_cache(maxsize=1)
def _get_tavily():
    tavily_search = TavilySearchAPIWrapper()
    return TavilySearchResults(api_wrapper=tavily_search, name="search_tavily")


@lru_cache(maxsize=1)
def _get_tavily_answer():
    tavily_search = TavilySearchAPIWrapper()
    return _TavilyAnswer(api_wrapper=tavily_search, name="search_tavily_answer")


@lru_cache(maxsize=1)
def _get_connery_actions():
    connery_service = ConneryService()
    connery_toolkit = ConneryToolkit.create_instance(connery_service)
    tools = connery_toolkit.get_tools()
    return tools


@lru_cache(maxsize=1)
def _get_dalle_tools():
    return Tool(
        "Dall-E-Image-Generator",
        DallEAPIWrapper(size="1024x1024", quality="hd").run,
        "A wrapper around OpenAI DALL-E API. Useful for when you need to generate images from a text description. Input should be an image description.",
    )


@lru_cache(maxsize=1)
def _get_issue_tree():
    return StructuredTool(
        name="fill_issue_tree",
        func=fill_issue_tree,
        description="Fill the issue tree framework.",
        args_schema=IssueTreeInput,
    )


class IssueTree(BaseTool):
    type: Literal[AvailableTools.ISSUE_TREE] = AvailableTools.ISSUE_TREE
    name: Literal["Issue Tree"] = "Issue Tree"
    description: Literal[
        "Fill the issue tree framework."
    ] = "Fill the issue tree framework."


def fill_issue_tree(**kwargs):
    """
    Fill the issue tree framework.
    """
    # Construct the IssueTreeInput model from keyword arguments
    input_model = IssueTreeInput(**kwargs)
    # Implement the function logic here
    # For now, we'll just return the input for demonstration purposes
    return input_model.dict()


class ApproachStep(BaseModel):
    mainStep: str = Field(
        description="The name of the main step in the approach."
    )
    subSteps: List[str] = Field(
        description="A list of substeps for the corresponding main step."
    )


class StrategicApproachInput(BaseModel):
    name: str = Field(
        description="Unique title for the assessment with 2-4 words, different than previous assessments",
        pattern=r"^\S+(?:\s+\S+){1,3}$"  # Ensures 2-4 words
    )
    type: str = Field(
        description="Identifies the structure as describing an approach.",
        pattern="^APPROACH$"
    )
    data: List[ApproachStep] = Field(
        description="An array of main steps, each with its substeps."
    )


def fill_strategic_approach(**kwargs):
    """
    Fill strategic approach framework.
    """
    # Construct the StrategicApproachInput model from keyword arguments
    input_model = StrategicApproachInput(**kwargs)
    # Implement the function logic here
    # For now, we'll just return the input for demonstration purposes
    return input_model.dict()


@lru_cache(maxsize=1)
def _get_strategic_approach():
    return StructuredTool(
        name="fill_strategic_approach",
        func=fill_strategic_approach,
        description="Fill strategic approach framework.",
        args_schema=StrategicApproachInput,
    )


class StrategicApproach(BaseTool):
    type: Literal[AvailableTools.STRATEGIC_APPROACH] = AvailableTools.STRATEGIC_APPROACH
    name: Literal["Strategic Approach"] = "Strategic Approach"
    description: Literal[
        "Fill strategic approach framework."
    ] = "Fill strategic approach framework."


class DecisionNode(BaseModel):
    nodeLabel: str = Field(
        description="The first question about a condition among a series of questions towards making the desired decision."
    )
    linkLabel: Optional[str] = Field(
        description="The label describing the alternative conditions leading to this node."
    )
    recommended: Optional[bool] = Field(
        description="For terminal nodes, indicates if this path is recommended."
    )
    children: Optional[List['DecisionNode']] = Field(
        default_factory=list,
        description="An array of child nodes representing subsequent questions about conditions."
    )

DecisionNode.update_forward_refs()

class DecisionTreeInput(BaseModel):
    name: str = Field(
        description="Unique title for the assessment with 2-4 words",
        pattern=r"^\S+(?:\s+\S+){1,3}$"  # Ensure consistency
    )
    type: str = Field(
        description="Identifies the type of the structure as a decision tree.",
        pattern="^DECISION_TREE$"
    )
    data: DecisionNode = Field(
        description="The root node of the decision tree, detailing the decision process."
    )

def fill_decision_tree(**kwargs):
    """
    Fill the decision tree framework.
    """
    input_model = DecisionTreeInput(**kwargs)
    return input_model.dict()

@lru_cache(maxsize=1)
def _get_decision_tree():
    return StructuredTool(
        name="fill_decision_tree",
        func=fill_decision_tree,
        description="Fill the decision tree framework.",
        args_schema=DecisionTreeInput,
    )

class DecisionTree(BaseTool):
    type: Literal[AvailableTools.DECISION_TREE] = AvailableTools.DECISION_TREE
    name: Literal["Decision Tree"] = "Decision Tree"
    description: Literal[
        "Fill the decision tree framework."
    ] = "Fill the decision tree framework."


class AxisLabels(BaseModel):
    xLabels: str = Field(
        description="Label for the X axis representing one criteria."
    )
    yLabels: str = Field(
        description="Label for the Y axis representing another criteria."
    )

class MatrixOption(BaseModel):
    x: float = Field(
        description="The X coordinate of the option out of 5."
    )
    y: float = Field(
        description="The Y coordinate of the option out of 5."
    )
    name: str = Field(
        description="The name of the option."
    )
    recommended: bool = Field(
        description="Indicates if the option is recommended."
    )

class RisksRewardsData(BaseModel):
    axisLabels: AxisLabels = Field(
        description="Labels for the X and Y axes of the matrix."
    )
    options: List[MatrixOption] = Field(
        description="Array of options plotted on the matrix."
    )

class RisksRewardsInput(BaseModel):
    name: str = Field(
        description="Unique title for the assessment with 2-4 words, different than previous assessments",
        pattern=r"^\S+(?:\s+\S+){1,3}$"  # Adjusted to 2-4 words
    )
    type: str = Field(
        description="Specifies the matrix type as risks and rewards.",
        pattern="^RISKS_REWARDS$"
    )
    data: RisksRewardsData = Field(
        description="The matrix data containing axis labels and plotted options."
    )

def fill_risks_rewards(**kwargs):
    """
    Fill the risks rewards analysis framework.
    """
    input_model = RisksRewardsInput(**kwargs)
    return input_model.dict()

@lru_cache(maxsize=1)
def _get_risks_rewards():
    return StructuredTool(
        name="fill_risks_rewards",
        func=fill_risks_rewards,
        description="Fill the risks rewards analysis framework.",
        args_schema=RisksRewardsInput,
    )

class RisksRewards(BaseTool):
    type: Literal[AvailableTools.RISKS_REWARDS] = AvailableTools.RISKS_REWARDS
    name: Literal["Risks Rewards"] = "Risks Rewards"
    description: Literal[
        "Fill the risks rewards analysis framework."
    ] = "Fill the risks rewards analysis framework."

class CriterionEvaluation(BaseModel):
    criterion: str = Field(
        description="The name of the criterion."
    )
    evaluation: str = Field(
        description="The evaluation of the option for the criterion (qualitative or quantitative)."
    )

class CostBenefitOption(BaseModel):
    name: str = Field(
        description="The name of the option."
    )
    costs: List[CriterionEvaluation] = Field(
        description="The costs associated with the option."
    )
    benefits: List[CriterionEvaluation] = Field(
        description="The benefits associated with the option."
    )
    recommended: bool = Field(
        description="Indicates whether this option is recommended."
    )

class CostBenefitData(BaseModel):
    options: List[CostBenefitOption] = Field(
        description="An array of options, each with its own set of costs and benefits."
    )

class CostBenefitInput(BaseModel):
    name: str = Field(
        description="Unique title for the assessment with 3-4 words, different than previous assessments"
    )
    type: str = Field(
        description="The type of analysis.",
        pattern="^COST_BENEFIT$"
    )
    data: CostBenefitData = Field(
        description="The cost-benefit analysis data containing options with their costs and benefits."
    )

def fill_cost_benefits_analysis(**kwargs):
    """
    Fill the cost vs benefits analysis framework, deciding on the cost and benefit 
    criteria and using them to evaluate multiple options.
    """
    input_model = CostBenefitInput(**kwargs)
    return input_model.dict()

@lru_cache(maxsize=1)
def _get_cost_benefits():
    return StructuredTool(
        name="fill_cost_benefits_analysis",
        func=fill_cost_benefits_analysis,
        description="Fill the cost vs benefits analysis framework, deciding on the cost and benefit criteria and using them to evaluate multiple options.",
        args_schema=CostBenefitInput,
    )

class CostBenefits(BaseTool):
    type: Literal[AvailableTools.COST_BENEFITS] = AvailableTools.COST_BENEFITS
    name: Literal["Cost Benefits Analysis"] = "Cost Benefits Analysis"
    description: Literal[
        "Fill the cost vs benefits analysis framework."
    ] = "Fill the cost vs benefits analysis framework."

class ProsConsOption(BaseModel):
    name: str = Field(
        description="The name of the option."
    )
    pros: List[str] = Field(
        description="List of advantages or positive aspects of this option."
    )
    cons: List[str] = Field(
        description="List of disadvantages or negative aspects of this option."
    )
    recommended: bool = Field(
        description="Indicates whether this option is recommended."
    )

class ProsConsData(BaseModel):
    options: List[ProsConsOption] = Field(
        description="Generate as many options as possible for the best assessment of the problem."
    )

class ProsConsInput(BaseModel):
    name: str = Field(
        description="Unique title for the assessment with 3-4 words, different than previous assessments"
    )
    type: str = Field(
        description="The type of analysis.",
        pattern="^PROS_CONS$"
    )
    data: ProsConsData = Field(
        description="The pros-cons analysis data containing options with their advantages and disadvantages."
    )

def fill_pros_cons(**kwargs):
    """
    Fill in the Pros Cons analysis framework.
    """
    input_model = ProsConsInput(**kwargs)
    return input_model.dict()

@lru_cache(maxsize=1)
def _get_pros_cons():
    return StructuredTool(
        name="fill_pros_cons",
        func=fill_pros_cons,
        description="Fill in the Pros Cons analysis framework.",
        args_schema=ProsConsInput,
    )

class ProsCons(BaseTool):
    type: Literal[AvailableTools.PROS_CONS] = AvailableTools.PROS_CONS
    name: Literal["Pros Cons Analysis"] = "Pros Cons Analysis"
    description: Literal[
        "Fill in the Pros Cons analysis framework."
    ] = "Fill in the Pros Cons analysis framework."


class RootCauseNode(BaseModel):
    nodeLabel: str = Field(
        description="Label for this node in the root cause analysis"
    )
    children: Optional[List['RootCauseNode']] = Field(
        default_factory=list,
        description="Child nodes representing deeper causes"
    )

RootCauseNode.update_forward_refs()

class RootCauseData(BaseModel):
    nodeLabel: str = Field(
        description="Root node of the root cause analysis"
    )
    children: List[RootCauseNode] = Field(
        description="Child nodes of the root cause"
    )

class RootCauseInput(BaseModel):
    name: str = Field(
        description="Unique title for the assessment with 3-4 words, different than previous assessments"
    )
    type: str = Field(
        description="The type of analysis.",
        pattern="^ROOT_CAUSE$"
    )
    data: RootCauseData = Field(
        description="The root cause analysis data containing the problem and its causes."
    )

def fill_root_cause_analysis(**kwargs):
    """
    Fill in the Root Cause Analysis framework.
    """
    input_model = RootCauseInput(**kwargs)
    return input_model.dict()

@lru_cache(maxsize=1)
def _get_root_cause():
    return StructuredTool(
        name="fill_root_cause_analysis",
        func=fill_root_cause_analysis,
        description="Fill in the Root Cause Analysis framework.",
        args_schema=RootCauseInput,
    )

class RootCause(BaseTool):
    type: Literal[AvailableTools.ROOT_CAUSE] = AvailableTools.ROOT_CAUSE
    name: Literal["Root Cause Analysis"] = "Root Cause Analysis"
    description: Literal[
        "Fill in the Root Cause Analysis framework."
    ] = "Fill in the Root Cause Analysis framework."

# SWOT Analysis Implementation
class SwotData(BaseModel):
    strengths: List[str] = Field(
        description="List of strengths."
    )
    weaknesses: List[str] = Field(
        description="List of weaknesses."
    )
    opportunities: List[str] = Field(
        description="List of opportunities."
    )
    threats: List[str] = Field(
        description="List of threats."
    )

class SwotInput(BaseModel):
    name: str = Field(
        description="Unique title for the assessment with 3-4 words, different than previous assessments"
    )
    type: str = Field(
        description="Specifies the type of document as SWOT.",
        pattern="^SWOT$"
    )
    data: SwotData = Field(
        description="The SWOT analysis data containing strengths, weaknesses, opportunities, and threats."
    )

def fill_swot_analysis(**kwargs):
    """
    Fill SWOT Analysis framework.
    """
    input_model = SwotInput(**kwargs)
    return input_model.dict()

@lru_cache(maxsize=1)
def _get_swot():
    return StructuredTool(
        name="fill_swot_analysis",
        func=fill_swot_analysis,
        description="Fill SWOT Analysis framework.",
        args_schema=SwotInput,
    )

class Swot(BaseTool):
    type: Literal[AvailableTools.SWOT] = AvailableTools.SWOT
    name: Literal["SWOT Analysis"] = "SWOT Analysis"
    description: Literal[
        "Fill SWOT Analysis framework."
    ] = "Fill SWOT Analysis framework."

# Multi-Criteria Scoring Implementation
class MultiCriteriaInput(BaseModel):
    name: str = Field(
        description="Unique title for the assessment with 3-4 words, different than previous assessments"
    )
    type: str = Field(
        description="The type of analysis.",
        pattern="^MULTI_CRITERIA$"
    )
    options: List[str] = Field(
        description="The options."
    )
    criteria: List[str] = Field(
        description="Type of criteria e.g. Performance, Affordability etc."
    )
    scores: List[List[float]] = Field(
        description="List of scores out of 10 for each criteria. The length must be equal to number of 'options'"
    )
    recommended: str = Field(
        description="Exact name of the recommended option. Critical to include, this must always be filled correctly with the recommended option's exact name."
    )

def fill_multi_criteria_scoring(**kwargs):
    """
    Fill in the Multi criteria scoring framework.
    """
    input_model = MultiCriteriaInput(**kwargs)
    return input_model.dict()

@lru_cache(maxsize=1)
def _get_multi_criteria():
    return StructuredTool(
        name="fill_multi_criteria_scoring",
        func=fill_multi_criteria_scoring,
        description="Fill in the Multi criteria scoring framework. Use as many options as possible",
        args_schema=MultiCriteriaInput,
    )

class MultiCriteria(BaseTool):
    type: Literal[AvailableTools.MULTI_CRITERIA] = AvailableTools.MULTI_CRITERIA
    name: Literal["Multi-Criteria Scoring"] = "Multi-Criteria Scoring"
    description: Literal[
        "Fill in the Multi criteria scoring framework."
    ] = "Fill in the Multi criteria scoring framework."

# Two by Two Matrix Implementation
class TwoByTwoAxisLabels(BaseModel):
    xLabels: str = Field(
        description="The label for the X axis, representing the first criteria."
    )
    yLabels: str = Field(
        description="The label for the Y axis, representing the second criteria."
    )

class TwoByTwoOption(BaseModel):
    x: float = Field(
        description="The X coordinate of the option on the matrix out of 5."
    )
    y: float = Field(
        description="The Y coordinate of the option on the matrix out of 5."
    )
    name: str = Field(
        description="The name of the option."
    )
    recommended: bool = Field(
        description="Indicates whether the option is recommended."
    )

class TwoByTwoData(BaseModel):
    axisLabels: TwoByTwoAxisLabels = Field(
        description="Labels for the X and Y axes of the matrix."
    )
    options: List[TwoByTwoOption] = Field(
        description="A list of options plotted on the 2x2 matrix."
    )

class TwoByTwoInput(BaseModel):
    name: str = Field(
        description="Unique title for the assessment with 3-4 words, different than previous assessments"
    )
    type: str = Field(
        description="The type of matrix, indicating a two-by-two matrix.",
        pattern="^TWO_BY_TWO$"
    )
    data: TwoByTwoData = Field(
        description="The matrix data containing axis labels and plotted options."
    )

def fill_two_by_two(**kwargs):
    """
    Fill Two by Two Matrix framework.
    """
    input_model = TwoByTwoInput(**kwargs)
    return input_model.dict()

@lru_cache(maxsize=1)
def _get_two_by_two():
    return StructuredTool(
        name="fill_two_by_two",
        func=fill_two_by_two,
        description="Fill Two by Two Matrix framework.",
        args_schema=TwoByTwoInput,
    )

class TwoByTwo(BaseTool):
    type: Literal[AvailableTools.TWO_BY_TWO] = AvailableTools.TWO_BY_TWO
    name: Literal["Two by Two Matrix"] = "Two by Two Matrix"
    description: Literal[
        "Fill Two by Two Matrix framework."
    ] = "Fill Two by Two Matrix framework."

# Morphological Box Implementation
class MorphBoxOption(BaseModel):
    name: str = Field(
        description="The name of the option."
    )
    selected: bool = Field(
        description="Indicates whether this option is selected."
    )

class MorphBoxCategory(BaseModel):
    header: str = Field(
        description="The category or aspect of the strategy."
    )
    options: List[MorphBoxOption] = Field(
        description="The options available within this category."
    )

class MorphBoxInput(BaseModel):
    name: str = Field(
        description="Unique title for the assessment with 3-4 words, different than previous assessments"
    )
    type: str = Field(
        description="The type of strategy document.",
        pattern="^MORPH_BOX$"
    )
    data: List[MorphBoxCategory] = Field(
        description="The list of strategy aspects and options."
    )

def fill_morphological_box(**kwargs):
    """
    Fill in the morphological box framework.
    """
    input_model = MorphBoxInput(**kwargs)
    return input_model.dict()

@lru_cache(maxsize=1)
def _get_morph_box():
    return StructuredTool(
        name="fill_morphological_box",
        func=fill_morphological_box,
        description="Fill in the morphological box framework.",
        args_schema=MorphBoxInput,
    )

class MorphBox(BaseTool):
    type: Literal[AvailableTools.MORPH_BOX] = AvailableTools.MORPH_BOX
    name: Literal["Morphological Box"] = "Morphological Box"
    description: Literal[
        "Fill in the morphological box framework."
    ] = "Fill in the morphological box framework."

# Matrix Analysis Implementation
class MatrixContent(BaseModel):
    columnHeader: str = Field(
        description="The name of the entity or category being evaluated."
    )
    rows: Dict[str, str] = Field(
        description="Evaluation data for each row header, where keys are row headers and values are the corresponding data.",
        default_factory=dict  # Ensure rows is always initialized
    )

    def __init__(self, **data):
        # Convert any extra fields into the rows dictionary
        rows_data = {}
        for key, value in data.items():
            if key != 'columnHeader':
                rows_data[key] = value
        if rows_data:
            data['rows'] = rows_data
        super().__init__(**data)

    class Config:
        # Allow dynamic field names
        extra = "allow"

class MatrixData(BaseModel):
    columnsDefinition: str = Field(
        description="Description of the columns in the matrix."
    )
    rowsDefinition: str = Field(
        description="Description of the rows in the matrix."
    )
    rowHeaders: List[str] = Field(
        description="List of row headers representing the criteria or attributes to be evaluated."
    )
    content: List[MatrixContent] = Field(
        description="Content for each entity or category being evaluated, organized by column headers.",
        default_factory=list
    )

    class Config:
        extra = "forbid"

class MatrixInput(BaseModel):
    name: str = Field(
        description="Insert a 2-4 word title for the matrix analysis."
    )
    type: str = Field(
        description="The type of analysis document.",
        pattern="^MATRIX_ANALYSIS$"
    )
    data: MatrixData = Field(
        description="The matrix analysis data containing definitions, headers, and content."
    )

def fill_matrix_analysis(**kwargs):
    """
    Generates a matrix analysis for comparing various entities against multiple criteria.
    """
    input_model = MatrixInput(**kwargs)
    return input_model.dict()

@lru_cache(maxsize=1)
def _get_matrix():
    return StructuredTool(
        name="fill_matrix_analysis",
        func=fill_matrix_analysis,
        description="Generates a matrix analysis for comparing various entities against multiple criteria.",
        args_schema=MatrixInput,
    )

class Matrix(BaseTool):
    type: Literal[AvailableTools.MATRIX] = AvailableTools.MATRIX
    name: Literal["Matrix Analysis"] = "Matrix Analysis"
    description: Literal[
        "Generates a matrix analysis for comparing various entities against multiple criteria."
    ] = "Generates a matrix analysis for comparing various entities against multiple criteria."

# Table Analysis Implementation
class TableData(BaseModel):
    headers: List[str] = Field(
        description="List of column headers for the table."
    )
    rows: List[Dict[str, str]] = Field(
        description=(
            "An array of objects, where each object represents a row in the table. "
            "Each row should align with the headers provided."
        )
    )

class TableAnalysisInput(BaseModel):
    name: str = Field(
        description="A 2-4 word title for the table analysis.",
        pattern=r"^\S+(?:\s+\S+){1,3}$"  # Ensures 2-4 words
    )
    type: str = Field(
        description="The type of analysis document.",
        pattern="^TABLE_ANALYSIS$"
    )
    data: TableData = Field(
        description="The table analysis data containing headers and rows."
    )

def fill_table_analysis(**kwargs):
    """
    Creates a detailed and specific table analysis.
    """
    input_model = TableAnalysisInput(**kwargs)
    # Here you can add any processing logic if needed
    return input_model.dict()

@lru_cache(maxsize=1)
def _get_table_analysis():
    return StructuredTool(
        name="fill_table_analysis",
        func=fill_table_analysis,
        description=(
            "Creates a detailed and specific table analysis. "
            "The analysis includes a list of headers and corresponding rows, "
            "allowing for a structured comparison of various entities."
        ),
        args_schema=TableAnalysisInput,
    )

class TableAnalysis(BaseTool):
    type: Literal[AvailableTools.TABLE_ANALYSIS] = AvailableTools.TABLE_ANALYSIS
    name: Literal["Table Analysis"] = "Table Analysis"
    description: Literal[
        "Creates a detailed and specific table analysis."
    ] = "Creates a detailed and specific table analysis."

# @lru_cache(maxsize=1)
# def _get_python_repl():
#     python_repl = PythonREPL()
#     return StructuredTool(
#         name="python_repl",
#         description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
#         func=python_repl.run,
#         args_schema=PythonREPLInput
#     )

# class PythonRepl(BaseTool):
#     type: Literal[AvailableTools.PYTHON_REPL] = AvailableTools.PYTHON_REPL
#     name: Literal["Python REPL"] = "Python REPL"
#     description: Literal[
#         "Execute Python commands in a REPL environment. Use print() to see output."
#     ] = "Execute Python commands in a REPL environment. Use print() to see output."

class AzureDynamicSessions(BaseTool):
    type: Literal[AvailableTools.AZURE_DYNAMIC_SESSIONS] = AvailableTools.AZURE_DYNAMIC_SESSIONS
    name: Literal["Azure Dynamic Sessions"] = "Azure Dynamic Sessions"
    description: Literal[
        "Execute Python code in secure Azure Container Apps environment. Includes popular packages like NumPy, pandas, and scikit-learn."
    ] = "Execute Python code in secure Azure Container Apps environment. Includes popular packages like NumPy, pandas, and scikit-learn."

@lru_cache(maxsize=1)
def _get_azure_dynamic_sessions():
    # Note: This assumes POOL_MANAGEMENT_ENDPOINT is configured in environment variables
    pool_endpoint = os.getenv("AZURE_POOL_MANAGEMENT_ENDPOINT")
    if not pool_endpoint:
        raise ValueError("AZURE_POOL_MANAGEMENT_ENDPOINT environment variable not set")
    return SessionsPythonREPLTool(pool_management_endpoint=pool_endpoint)

TOOLS = {
    AvailableTools.CONNERY: _get_connery_actions,
    AvailableTools.DDG_SEARCH: _get_duck_duck_go,
    AvailableTools.ARXIV: _get_arxiv,
    AvailableTools.YOU_SEARCH: _get_you_search,
    AvailableTools.SEC_FILINGS: _get_sec_filings,
    AvailableTools.PRESS_RELEASES: _get_press_releases,
    AvailableTools.PUBMED: _get_pubmed,
    AvailableTools.TAVILY: _get_tavily,
    AvailableTools.WIKIPEDIA: _get_wikipedia,
    AvailableTools.TAVILY_ANSWER: _get_tavily_answer,
    AvailableTools.DALL_E: _get_dalle_tools,
    AvailableTools.ISSUE_TREE: _get_issue_tree,
    AvailableTools.STRATEGIC_APPROACH: _get_strategic_approach,
    AvailableTools.DECISION_TREE: _get_decision_tree,
    AvailableTools.RISKS_REWARDS: _get_risks_rewards,
    AvailableTools.COST_BENEFITS: _get_cost_benefits,
    AvailableTools.PROS_CONS: _get_pros_cons,
    AvailableTools.ROOT_CAUSE: _get_root_cause,
    AvailableTools.SWOT: _get_swot,
    AvailableTools.MULTI_CRITERIA: _get_multi_criteria,
    AvailableTools.TWO_BY_TWO: _get_two_by_two,
    AvailableTools.MORPH_BOX: _get_morph_box,
    AvailableTools.MATRIX: _get_matrix,
    AvailableTools.TABLE_ANALYSIS: _get_table_analysis,
    # AvailableTools.PYTHON_REPL: _get_python_repl,
    AvailableTools.AZURE_DYNAMIC_SESSIONS: _get_azure_dynamic_sessions,
}
