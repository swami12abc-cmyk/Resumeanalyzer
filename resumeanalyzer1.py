import streamlit as st
import PyPDF2
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
import os

# Load environment variables
load_dotenv()

# Initialize LLM
llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.1,
)

# Resume Analysis Agent: extracts candidate name, score, and short analysis
resume_agent = Agent(
    role="Resume Analysis Agent",
    goal="Analyze resume content and provide candidate name, suitability score, and short analysis",
    backstory="An AI that evaluates resumes for a job description and outputs candidate name, score, and short analysis",
    llm=llm
)

# Score Table Agent: generates final formatted table
score_table_agent = Agent(
    role="Score Table Agent",
    goal="Create a formatted table of resumes with candidate name, score, and short analysis",
    backstory="An AI that organizes and formats candidate scores and analysis in a table",
    llm=llm
)

st.title("CrewAI Resume Analyzer")

# Upload multiple resumes and job description
uploaded_resumes = st.file_uploader(
    "Upload Resumes (PDF or TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)
uploaded_job_desc = st.file_uploader("Upload Job Description (TXT)", type=["txt"])

if uploaded_resumes and uploaded_job_desc:
    job_text = uploaded_job_desc.read().decode("utf-8")

    if st.button("Analyze"):
        with st.spinner("Analyzing resumes..."):
            resume_data_list = []

            for uploaded_resume in uploaded_resumes:
                # Read resume content
                if uploaded_resume.type == "application/pdf":
                    pdf_reader = PyPDF2.PdfReader(uploaded_resume)
                    resume_text = ""
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            resume_text += page_text + "\n"
                else:
                    resume_text = uploaded_resume.read().decode("utf-8")

                # Task to extract candidate name, score, and short analysis based on resume text and JD
                resume_task = Task(
                    description=f"""
                    Analyze the following resume content:
                    '''{resume_text}'''

                    Based on this job description:
                    '''{job_text}'''

                    Provide:
                    1. Candidate Name: Extract from resume content or filename if not present.
                    2. Suitability Score: Rate from 0 to 10 based on candidate's match to the job description.
                    3. Short Analysis: Brief summary of strengths and gaps of the candidate.

                    Output format:
                    Candidate Name: <name>, Score: <score>, Analysis: <short analysis>
                    """,
                    agent=resume_agent,
                    expected_output="Candidate name, score, and short analysis"
                )

                crew = Crew(
                    agents=[resume_agent],
                    tasks=[resume_task],
                    verbose=False
                )

                # Get output
                crew_result = crew.kickoff()
                output = crew_result.tasks_output[0].raw
                resume_data_list.append(output)

            # Combine all resume outputs as context for score table
            combined_context = "\n".join(resume_data_list)

            # Task to create formatted table with three columns
            score_task = Task(
                description=f"""
                Create a table with three columns: 'Resume / Candidate Name', 'Score', and 'Analysis'
                based on the following candidate data:
                '''{combined_context}'''
                """,
                agent=score_table_agent,
                expected_output="Formatted table with name, score, and analysis",
                context=[]  # optionally could pass resume tasks for context
            )

            crew_table = Crew(
                agents=[score_table_agent],
                tasks=[score_task],
                verbose=False
            )

            table_result = crew_table.kickoff()

            st.subheader("Resume Score Table")
            st.text(table_result.tasks_output[0].raw)
