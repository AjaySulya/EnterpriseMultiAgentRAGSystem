"""

Router Agent : langgraph workflow  that classifies query and dispatches it to the correst specialist agent.

Graph:
         START - classify - [pdf agent | sql agent | web agent] - END .
"""


from typing import Annotated, Any, TypedDict

from langgraph.graph import START,StateGraph,END

from app.agents.pdf_agent import PDFAgent
from app.agents.sql_agent import SQLAgent
from app.agents.web_agent import WEBAgent

from app.utils.logger import get_logger


logger = get_logger(__name__)


"""State """

class AgentState(TypedDict):
    """Shared memory updates and passes between graph nodes."""
    
    query: str
    session_id: str
    route: str  # pdf,web,sql
    answer: str
    sources:list[dict]
    agent_used: str
    extra:dict[str,Any]



# ─── Routing keywords ────────────────────────────────────────────────────────
 
_PDF_KEYWORDS = {
    "document", "pdf", "file", "report", "uploaded", "page",
    "text", "paragraph", "section", "article",
}
_SQL_KEYWORDS = {
    "database", "table", "sql", "record", "row", "count",
    "how many", "list all", "show me", "users", "documents",
    "chat history", "registered",
}
_WEB_KEYWORDS = {
    "http", "https", "www.", "website", "webpage", "url",
    "site", "scrape", "page at", "link",
}


def _classify(query:str) -> str:
    
    """Rule-based classifier Returns 'pdf' , 'sql' 'web' .
    Fall back to pdf when ambiguous."""
    
    lower = query.lower()
    
    
    if any(k in lower for k in _WEB_KEYWORDS):
        return "web"
    
    
    sql_score = sum(1 for k in _SQL_KEYWORDS if k in lower)
    
    pdf_score = sum(1 for k in _PDF_KEYWORDS if k in lower)
    
    if sql_score > pdf_score:
        return "sql"
    
    return "pdf"



# state nodes 

async def classify_node(state:AgentState) -> AgentState:
    """
    Classify the query and  set the routing key
    
    """
    route = _classify(state["query"])
    logger.info("Classify the query",query=state["query"][:60],route = route)
    return {**state,"route":route}


async def pdf_node(state:AgentState) -> AgentState:
    """Invoke the pdf RAG agent."""
    
    agent = PDFAgent()
    result = await agent.answer(state['query'])
    
    return {
        **state,
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "agent_used": "pdf_agent",
    }
    
    
async def sql_node(state:AgentState) -> AgentState:
    """Invoke the sql agent"""
    
    db = state["extra"].get("db")
    
    if db is None:
        return {**state, "answer": "DataBase session unavailable.","agent_used":"sql_agent"}
    
    agent = SQLAgent(db=db)

    result = await agent.answer(state["query"])
    return {
        **state,
        "answer": result["answer"],
        "sources": [{"sql": result.get("sql", "")}],
        "agent_used": "sql_agent",
        "extra": {**state["extra"], "rows": result.get("rows", [])},
    }    
    
    
    
async def web_node(state:AgentState) -> AgentState:
    """Invoke the web agent"""
    
    agent = WEBAgent()
    
    result = await agent.answer(state["query"])

    return {
        **state,
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "agent_used": "web_agent",
    }
    
    
def route_selector(state:AgentState) -> str:
    """LangGraph conditional edge - maps route - node name."""
    
    return state['route']



def build_graph() -> StateGraph:
    """Construct and compile mult-agent  LangGraph workflow."""
    
    builder = StateGraph(AgentState)
    
    builder.add_node("classifier",classify_node)
    builder.add_node("pdf",pdf_node)
    builder.add_node("web",web_node)
    builder.add_node("sql",sql_node)
    
    builder.add_edge(START,"classifier")
    builder.add_conditional_edges(
            "classifier",
            route_selector,
            {"pdf": "pdf", "sql": "sql", "web": "web"},
            )
    builder.add_edge("pdf", END)
    builder.add_edge("sql", END)
    builder.add_edge("web", END)
    
    return  builder.compile()
    
    
# Module-level compiled graph (singleton)
rag_graph = build_graph()

async def run_rag_pipeline(
        query: str,
        session_id: str = "default",
        db: Any = None,
        ) -> AgentState:
        """
        Public entry point — run the full multi-agent pipeline.
 
        Args:
        query:      User question.
        session_id: Conversation session ID.
        db:         AsyncSession (required for SQL agent).
 
        Returns:
            Final AgentState dict.
        """
        initial_state: AgentState = {
            "query": query,
            "session_id": session_id,
            "route": "",
            "answer": "",
            "sources": [],
            "agent_used": "",
            "extra": {"db": db},
            }
        
        result = await rag_graph.ainvoke(initial_state)
        return result
        
    
    