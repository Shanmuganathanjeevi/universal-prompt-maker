import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# --- 1. Pydantic Schema for the Triage Agent ---
class TriageQuestions(BaseModel):
    question_1: str = Field(description="The most critical question to narrow down the core context or subject matter of the user's topic.")
    question_2: str = Field(description="A question focusing on the desired format, tone, style, or constraints.")
    question_3: str = Field(description="A question focusing on the target audience, the ultimate goal, or the platform where this will be used.")

# --- 2. App Config & Phase State Management ---
st.set_page_config(page_title="Universal Prompt Maker", page_icon="🧠", layout="centered")

# Initialize Auth State
if "api_key_valid" not in st.session_state:
    st.session_state.api_key_valid = False
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gemini-3.5-flash-lite"

# Initialize Multi-Step Workflow State
if "workflow_phase" not in st.session_state:
    st.session_state.workflow_phase = 1  # Phase 1: Seed, 2: Questions, 3: Final Prompt
if "seed_topic" not in st.session_state:
    st.session_state.seed_topic = ""
if "dynamic_questions" not in st.session_state:
    st.session_state.dynamic_questions = None
if "final_master_prompt" not in st.session_state:
    st.session_state.final_master_prompt = ""

# --- 3. The Robust Sidebar Gatekeeper ---
with st.sidebar:
    st.header("⚙️ System Config")
    
    provider = st.selectbox("Select AI Provider", ["Gemini", "ChatGPT", "Claude"])
    
    if provider != "Gemini":
        st.info(f"{provider} integration is coming soon! Please select Gemini to continue.")
        st.session_state.api_key_valid = False
    else:
        selected_model = st.selectbox(
            "Select Model", 
            ["gemini-3.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]
        )
        
        api_input = st.text_input(
            "Gemini API Key", 
            type="password", 
            placeholder="Paste your API key here...",
            value=st.session_state.api_key
        )
        
        if api_input.strip():
            st.session_state.api_key = api_input.strip()
            st.session_state.api_key_valid = True
            st.session_state.selected_model = selected_model
            st.success("✅ System Unlocked")
        else:
            st.session_state.api_key_valid = False
            st.info("Awaiting API Key...")
            st.markdown("[Get a free Google API key here](https://aistudio.google.com/app/apikey)")
            
    st.divider()
    
    if st.button("🔄 Reset Workflow", use_container_width=True):
        st.session_state.workflow_phase = 1
        st.session_state.seed_topic = ""
        st.session_state.dynamic_questions = None
        st.session_state.final_master_prompt = ""
        st.rerun()

# --- 4. Lock the App ---
st.title("🧠 Universal Prompt Maker")
st.write("I build perfect, high-context prompts for any LLM. Give me a raw topic, I will ask you 3 clarifying questions, and then I will engineer the ultimate master prompt.")

if not st.session_state.api_key_valid:
    st.warning("👈 Please select a provider and authenticate in the sidebar to begin.")
    st.stop()

# Initialize the GenAI client
client = genai.Client(api_key=st.session_state.api_key)

# --- 5. Phase 1: The Seed (Triage Agent Trigger) ---
if st.session_state.workflow_phase == 1:
    with st.form("seed_form"):
        seed_topic = st.text_input(
            "What do you want to create, solve, or write?", 
            placeholder="e.g., A Python script for data analysis, a marketing email, a business proposal..."
        )
        submitted = st.form_submit_button("Analyze & Generate Questions")
        
        if submitted:
            if not seed_topic.strip():
                st.warning("Please provide a topic.")
                st.stop()
                
            st.session_state.seed_topic = seed_topic
            
            with st.spinner(f"Triage Agent is analyzing your request using {st.session_state.selected_model}..."):
                try:
                    triage_prompt = f"""
                    You are an expert AI Prompt Engineer. The user wants to write a prompt for the following topic: '{st.session_state.seed_topic}'.
                    To write the most powerful, context-rich prompt possible for them, what 3 clarifying questions MUST you ask the user?
                    """
                    response = client.models.generate_content(
                        model=st.session_state.selected_model,
                        contents=triage_prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=TriageQuestions,
                            temperature=0.7,
                        ),
                    )
                    st.session_state.dynamic_questions = response.parsed
                    st.session_state.workflow_phase = 2
                    st.rerun()
                except Exception as e:
                    st.error(f"API Error: {e}")

# --- 6. Phase 2: The Interview (Data Collection) ---
elif st.session_state.workflow_phase == 2:
    st.info(f"**Topic:** {st.session_state.seed_topic}")
    st.subheader("Let's refine the context")
    
    with st.form("interview_form"):
        ans_1 = st.text_area(f"1. {st.session_state.dynamic_questions.question_1}")
        ans_2 = st.text_area(f"2. {st.session_state.dynamic_questions.question_2}")
        ans_3 = st.text_area(f"3. {st.session_state.dynamic_questions.question_3}")
        
        submitted = st.form_submit_button("Engineer Master Prompt")
        
        if submitted:
            with st.spinner("Master Prompt Engineer is synthesizing..."):
                try:
                    master_prompt_instruction = f"""
                    You are a Master Prompt Engineer. Your task is to write a single, highly optimized prompt 
                    that the user can copy and paste into ANY LLM (ChatGPT, Claude, Gemini) to get a perfect result.
                    
                    Base Topic: {st.session_state.seed_topic}
                    
                    User Context 1 ({st.session_state.dynamic_questions.question_1}): {ans_1}
                    User Context 2 ({st.session_state.dynamic_questions.question_2}): {ans_2}
                    User Context 3 ({st.session_state.dynamic_questions.question_3}): {ans_3}
                    
                    Instructions for you:
                    1. Act as the prompt engineer. Output ONLY the raw prompt text that the user will copy. Do not include introductory conversational text.
                    2. Use best practices: Assign a role to the AI, define the exact task, inject the user's context, set clear constraints, and specify the output format.
                    """
                    
                    master_response = client.models.generate_content(
                        model=st.session_state.selected_model,
                        contents=master_prompt_instruction,
                    )
                    
                    st.session_state.final_master_prompt = master_response.text
                    st.session_state.workflow_phase = 3
                    st.rerun()
                except Exception as e:
                    st.error(f"API Error: {e}")

# --- 7. Phase 3: The Final Output ---
elif st.session_state.workflow_phase == 3:
    st.success("✨ Your Master Prompt is ready!")
    st.write("Hover over the top right corner of the box below to copy this prompt, then paste it into any AI tool.")
    
    # st.code provides a native copy-to-clipboard button
    st.code(st.session_state.final_master_prompt, language="markdown")
    
    st.divider()
    if st.button("🚀 Start a New Prompt", type="primary"):
        st.session_state.workflow_phase = 1
        st.session_state.seed_topic = ""
        st.session_state.dynamic_questions = None
        st.session_state.final_master_prompt = ""
        st.rerun()