from langchain_core.tools import tool


@tool(parse_docstring=True)
def job_search_think_tool(reflection: str) -> str:
    """Tool for strategic reflection on job search progress and decision-making.

    Use this tool after each job search or information gathering step to analyze results 
    and plan next steps systematically. This creates a deliberate pause in the job search 
    workflow for quality decision-making.

    When to use:
    - After searching for job positions: What job opportunities did I find? Are they relevant?
    - After gathering job details: Do I have complete information (requirements, salary, benefits, company culture)?
    - Before deciding next steps: Do I have enough information to provide comprehensive job recommendations?
    - When assessing job matching: How well do the found positions match the user's job brief?
    - When evaluating search completeness: Have I covered different job sources, companies, or positions?
    - Before concluding: Can I provide actionable job search recommendations now?

    Reflection should address:
    1. Job information analysis - What specific job positions, companies, or opportunities have I found?
    2. Information completeness - Do I have key details: job requirements, salary range, location, company info, application process?
    3. Match assessment - How well do the found positions align with the user's job brief (skills, experience, preferences)?
    4. Gap identification - What crucial information is still missing (specific companies, salary data, market trends, application tips)?
    5. Quality evaluation - Do I have sufficient information to provide valuable job search recommendations?
    6. Strategic decision - Should I continue searching for more positions/information, or provide recommendations now?

    Args:
        reflection: Your detailed reflection on job search progress, findings, job matches, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"
