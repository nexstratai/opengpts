from enum import Enum
from functools import lru_cache
from typing import Optional, List, Dict

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.retriever import create_retriever_tool
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
from langchain_core.tools import Tool
from langchain_robocorp import ActionServerToolkit
from pydantic import ValidationError
from typing_extensions import TypedDict
from langchain.tools import StructuredTool
from langchain_core.documents import Document

from app.upload import vstore


class DDGInput(BaseModel):
    query: str = Field(description="search query to look up")


class ArxivInput(BaseModel):
    query: str = Field(description="search query to look up")


class PythonREPLInput(BaseModel):
    query: str = Field(description="python command to run")


class DallEInput(BaseModel):
    query: str = Field(description="image description to generate image from")


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


class ToolConfig(TypedDict):
    ...


class BaseTool(BaseModel):
    type: AvailableTools
    name: Optional[str]
    description: Optional[str]
    config: Optional[ToolConfig]
    multi_use: Optional[bool] = False


class ActionServerConfig(ToolConfig):
    url: str
    api_key: str


class ActionServer(BaseTool):
    type: AvailableTools = Field(AvailableTools.ACTION_SERVER, const=True)
    name: str = Field("Action Server by Sema4.ai", const=True)
    description: str = Field(
        (
            "Run AI actions with "
            "[Sema4.ai Action Server](https://github.com/Sema4AI/actions)."
        ),
        const=True,
    )
    config: ActionServerConfig
    multi_use: bool = Field(True, const=True)


class Connery(BaseTool):
    type: AvailableTools = Field(AvailableTools.CONNERY, const=True)
    name: str = Field("AI Action Runner by Connery", const=True)
    description: str = Field(
        (
            "Connect OpenGPTs to the real world with "
            "[Connery](https://github.com/connery-io/connery)."
        ),
        const=True,
    )


class DDGSearch(BaseTool):
    type: AvailableTools = Field(AvailableTools.DDG_SEARCH, const=True)
    name: str = Field("DuckDuckGo Search", const=True)
    description: str = Field(
        "Search the web with [DuckDuckGo](https://pypi.org/project/duckduckgo-search/).",
        const=True,
    )


class Arxiv(BaseTool):
    type: AvailableTools = Field(AvailableTools.ARXIV, const=True)
    name: str = Field("Arxiv", const=True)
    description: str = Field("Searches [Arxiv](https://arxiv.org/).", const=True)


class YouSearch(BaseTool):
    type: AvailableTools = Field(AvailableTools.YOU_SEARCH, const=True)
    name: str = Field("You.com Search", const=True)
    description: str = Field(
        "Uses [You.com](https://you.com/) search, optimized responses for LLMs.",
        const=True,
    )


class SecFilings(BaseTool):
    type: AvailableTools = Field(AvailableTools.SEC_FILINGS, const=True)
    name: str = Field("SEC Filings (Kay.ai)", const=True)
    description: str = Field(
        "Searches through SEC filings using [Kay.ai](https://www.kay.ai/).", const=True
    )


class PressReleases(BaseTool):
    type: AvailableTools = Field(AvailableTools.PRESS_RELEASES, const=True)
    name: str = Field("Press Releases (Kay.ai)", const=True)
    description: str = Field(
        "Searches through press releases using [Kay.ai](https://www.kay.ai/).",
        const=True,
    )


class PubMed(BaseTool):
    type: AvailableTools = Field(AvailableTools.PUBMED, const=True)
    name: str = Field("PubMed", const=True)
    description: str = Field(
        "Searches [PubMed](https://pubmed.ncbi.nlm.nih.gov/).", const=True
    )


class Wikipedia(BaseTool):
    type: AvailableTools = Field(AvailableTools.WIKIPEDIA, const=True)
    name: str = Field("Wikipedia", const=True)
    description: str = Field(
        "Searches [Wikipedia](https://pypi.org/project/wikipedia/).", const=True
    )


class Tavily(BaseTool):
    type: AvailableTools = Field(AvailableTools.TAVILY, const=True)
    name: str = Field("Search (Tavily)", const=True)
    description: str = Field(
        (
            "Uses the [Tavily](https://app.tavily.com/) search engine. "
            "Includes sources in the response."
        ),
        const=True,
    )


class TavilyAnswer(BaseTool):
    type: AvailableTools = Field(AvailableTools.TAVILY_ANSWER, const=True)
    name: str = Field("Search (short answer, Tavily)", const=True)
    description: str = Field(
        (
            "Uses the [Tavily](https://app.tavily.com/) search engine. "
            "This returns only the answer, no supporting evidence."
        ),
        const=True,
    )


class Retrieval(BaseTool):
    type: AvailableTools = Field(AvailableTools.RETRIEVAL, const=True)
    name: str = Field("Retrieval", const=True)
    description: str = Field("Look up information in uploaded files.", const=True)


class DallE(BaseTool):
    type: AvailableTools = Field(AvailableTools.DALL_E, const=True)
    name: str = Field("Generate Image (Dall-E)", const=True)
    description: str = Field(
        "Generates images from a text description using OpenAI's DALL-E model.",
        const=True,
    )


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
        regex="^ISSUE_TREE$"
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


def _get_action_server(**kwargs: ActionServerConfig):
    toolkit = ActionServerToolkit(
        url=kwargs["url"],
        api_key=kwargs["api_key"],
        additional_headers=kwargs.get("additional_headers", {}),
    )
    tools = toolkit.get_tools()
    return tools


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
    type: AvailableTools = Field(AvailableTools.ISSUE_TREE, const=True)
    name: str = Field("Issue Tree", const=True)
    description: str = Field("Fill the issue tree framework.", const=True)


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
        regex=r"^\S+(?:\s+\S+){1,3}$"  # Ensures 2-4 words
    )
    type: str = Field(
        description="Identifies the structure as describing an approach.",
        regex="^APPROACH$"
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
    type: AvailableTools = Field(AvailableTools.STRATEGIC_APPROACH, const=True)
    name: str = Field("Strategic Approach", const=True)
    description: str = Field("Fill strategic approach framework.", const=True)


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
        regex=r"^\S+(?:\s+\S+){1,3}$"  # Ensure consistency
    )
    type: str = Field(
        description="Identifies the type of the structure as a decision tree.",
        regex="^DECISION_TREE$"
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
    type: AvailableTools = Field(AvailableTools.DECISION_TREE, const=True)
    name: str = Field("Decision Tree", const=True)
    description: str = Field("Fill the decision tree framework.", const=True)


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
        regex=r"^\S+(?:\s+\S+){1,3}$"  # Adjusted to 2-4 words
    )
    type: str = Field(
        description="Specifies the matrix type as risks and rewards.",
        regex="^RISKS_REWARDS$"
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
    type: AvailableTools = Field(AvailableTools.RISKS_REWARDS, const=True)
    name: str = Field("Risks Rewards", const=True)
    description: str = Field("Fill the risks rewards analysis framework.", const=True)

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
        regex="^COST_BENEFIT$"
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
    type: AvailableTools = Field(AvailableTools.COST_BENEFITS, const=True)
    name: str = Field("Cost Benefits Analysis", const=True)
    description: str = Field("Fill the cost vs benefits analysis framework.", const=True)

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
        regex="^PROS_CONS$"
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
    type: AvailableTools = Field(AvailableTools.PROS_CONS, const=True)
    name: str = Field("Pros Cons Analysis", const=True)
    description: str = Field("Fill in the Pros Cons analysis framework.", const=True)


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
        regex="^ROOT_CAUSE$"
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
    type: AvailableTools = Field(AvailableTools.ROOT_CAUSE, const=True)
    name: str = Field("Root Cause Analysis", const=True)
    description: str = Field("Fill in the Root Cause Analysis framework.", const=True)

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
        regex="^SWOT$"
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
    type: AvailableTools = Field(AvailableTools.SWOT, const=True)
    name: str = Field("SWOT Analysis", const=True)
    description: str = Field("Fill SWOT Analysis framework.", const=True)

# Multi-Criteria Scoring Implementation
class MultiCriteriaInput(BaseModel):
    name: str = Field(
        description="Unique title for the assessment with 3-4 words, different than previous assessments"
    )
    type: str = Field(
        description="The type of analysis.",
        regex="^MULTI_CRITERIA$"
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
    type: AvailableTools = Field(AvailableTools.MULTI_CRITERIA, const=True)
    name: str = Field("Multi-Criteria Scoring", const=True)
    description: str = Field("Fill in the Multi criteria scoring framework.", const=True)

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
        regex="^TWO_BY_TWO$"
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
    type: AvailableTools = Field(AvailableTools.TWO_BY_TWO, const=True)
    name: str = Field("Two by Two Matrix", const=True)
    description: str = Field("Fill Two by Two Matrix framework.", const=True)

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
        regex="^MORPH_BOX$"
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
    type: AvailableTools = Field(AvailableTools.MORPH_BOX, const=True)
    name: str = Field("Morphological Box", const=True)
    description: str = Field("Fill in the morphological box framework.", const=True)

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
        regex="^MATRIX_ANALYSIS$"
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
    type: AvailableTools = Field(AvailableTools.MATRIX, const=True)
    name: str = Field("Matrix Analysis", const=True)
    description: str = Field("Generates a matrix analysis for comparing various entities against multiple criteria.", const=True)

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
        regex=r"^\S+(?:\s+\S+){1,3}$"  # Ensures 2-4 words
    )
    type: str = Field(
        description="The type of analysis document.",
        regex="^TABLE_ANALYSIS$"
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
    type: AvailableTools = Field(AvailableTools.TABLE_ANALYSIS, const=True)
    name: str = Field("Table Analysis", const=True)
    description: str = Field(
        "Creates a detailed and specific table analysis.",
        const=True
    )

TOOLS = {
    AvailableTools.ACTION_SERVER: _get_action_server,
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
}
