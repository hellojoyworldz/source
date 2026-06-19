import logging

from langgraph.graph import END, START, StateGraph

from backend.graph.state import StockAnalysisState
from backend.graph.nodes import (
    news_node,
    financial_node,
    technical_node,
    competitor_node,
    report_node,
)

logger = logging.getLogger(__name__)


def build_stock_graph():

    # 그래프 생성
    graph = StateGraph(StockAnalysisState)

    # 노드 등록
    graph.add_node("news_node", news_node)
    graph.add_node("financial_node", financial_node)
    graph.add_node("technical_node", technical_node)
    graph.add_node("competitor_node", competitor_node)
    graph.add_node("report_node", report_node)

    # 연결
    graph.add_edge(START, "news_node")
    graph.add_edge("news_node", "financial_node")
    graph.add_edge("financial_node", "technical_node")
    graph.add_edge("technical_node", "competitor_node")
    graph.add_edge("competitor_node", "report_node")
    graph.add_edge("report_node", END)

    # 실행
    compiled = graph.compile()
    logger.info("LangGraph 주식 분석 그래프 빌드 완료")

    return compiled
