REPORT_PROMPT = """
당신은 월가 애널리스트입니다.

회사명: 
{company_name}

최근뉴스:
{news}

재무정보: 
{financials}

기술적지표:
{technicals}

경쟁사:
{competitors}

다음 형식으로 분석하세요.
1. 기업 개요
2. 최근 뉴스 핵심 요약
3. 재무 상태 분석
4. 기술적 분석
5. 경쟁사 비교
6. 주요 리스크
7. 종합 의견

반드시 구체적인 수치를 인용하세요.

"""

INVESTOR_SENTIMENT_PROMPT = """다음 뉴스들이 전체적인 투자 심리를 분석하시오.

{news_text}

반드시 JSON형식으로 응답

{{
    "positive":0,
    "neutral":0,
    "negative":0,
}}
"""
