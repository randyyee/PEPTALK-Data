import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_experimental.agents.agent_toolkits import create_csv_agent, create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import AzureChatOpenAI

# from dotenv import load_dotenv # If deploying add variables in .env in RStudio Connect manually in order for the app to work!

data_library = {
    "Test Data: Vietnam PSNUxIM": "resources/Vietnam_PSNU_IM_FY22-24.csv",
    "OUxIM": "resources/OUxIM.csv"
}
narratives_path = "resources/NarrativesRaw__2024-05-14_.csv"

st.set_page_config(page_title="PEPTALK Data Demo", layout="wide")

with st.sidebar:
    st.subheader("PEPTALK: Chat with Data Demo")
    st.markdown(
        "This is an app to demo a PEPFAR data analysis chatbot use case. "
        "The default settings use Azure OpenAI. "
        "Select the dataset you want to use then ask your query. "
        "Data comes from PEPFAR Panorama (https://data.pepfar.gov/). "
        "This app has PSNUxIM and MER Narrative (all-time) datasets. "
    )

    user_data = st.selectbox("Select a data file",
    options=data_library.keys())


# Import data
data_path = data_library[user_data]
df = pd.read_csv(data_path, low_memory=False)
df1 = pd.read_csv(narratives_path, low_memory=False) 
df1 = df1.drop(df1.columns[1], axis=1)

# Set up llm for all the agents
azure_llm = AzureChatOpenAI(
    api_version=os.environ["AZURE_OPENAI_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    api_key=os.environ["AZURE_OPENAI_KEY"],
    temperature = 0
)

# Set up agents for the three use cases (the MSD ones will share one agent)
agent = create_pandas_dataframe_agent(
    azure_llm,
    df,
    verbose=True,
    allow_dangerous_code=True
)

agent1 = create_pandas_dataframe_agent(
    azure_llm,
    df1,
    verbose=True,
    allow_dangerous_code=True
)

# Set up the columns
chat_d, chat_v, chat_n = st.columns(3)

with chat_d:
    st.header("Chat with Your Data")
    with st.expander("Example queries", expanded=True):
        st.markdown(
            '''
            - The testing yield is HTS_TST_POS/HTS_TST. Using this calculation, can you figure out which mech_name has the lowest yield for period 2023 cumulative? Start by filtering for the standardizeddisaggregate "Total Numerator" then group by and summarize by mech_name before adding the yield.
            - Using the HTS_TST column, make a bar chart for the total number of tests performed by each psnu in 2022 qtr4? Start by filtering for the standardizeddisaggregate "Total Numerator" then group by psnu.
            - The linkage is calculated using TX_NEW/HTS_TST_POS. Using this calculation, can you figure out which mech_name has the lowest linkage for the period 2023 cumulative? Start by filtering for the standardizeddisaggregate "Total Numerator" then group by and summarize by mech_name before adding the linkage.
            - The linkage is calculated using TX_NEW/HTS_TST_POS. Using this calculation, can you compare the ratio in 2023 qtr4 between age_2019 using a statistical test? Be sure to group by and summarize after filtering and before calculating the linkage.
            '''
        )

    with st.expander("Dataset preview"):
        st.dataframe(df.head(), use_container_width=True)  # Show data

    user_query = st.text_area("Ask a question about the MER data.")

    if st.button("Submit", key = 1) and len(user_query) > 0:

        response = agent.invoke(user_query)
        st.write(response["output"])
        st.write(response)

with chat_v:
    st.header("Ask for a Data Visualization")
    with st.expander("Example queries", expanded=True):
        st.markdown(
            '''
            - Using the HTS_TST column, make a bar chart for the total number of tests performed by each psnu in 2022 qtr4? Start by filtering for the standardizeddisaggregate "Total Numerator" then group by psnu.
            - Using the HTS_TST_POS column, make a pie chart for the total number of positives found by each psnu in 2023 cumulative? Start by filtering for the standardizeddisaggregate "Total Numerator" then group by psnu.
            - Using the TX_NEW column, make a pie chart for the total number of new on treatment by each age_2019 and sex combination in 2023 cumulative? Start by filtering for the standardizeddisaggregate "Age/Sex/HIVStatus" then group by age_2019 and sex.
            '''
        )

    user_query1 = st.text_area("Make a viz request.")

    if st.button("Submit", key = 2) and len(user_query1) > 0:

        agent = create_pandas_dataframe_agent(
            azure_llm,
            df,
            verbose=True,
            allow_dangerous_code=True
        )

        plot_area = st.empty()
        response1 = agent.invoke(user_query1 + " Generate Python Code Script. The script should only include code.")
        st.write(response1)
        try:
            extracted_code = response1["output"].split('```python\n')[1].split('\n```')[0]
            primer_code = "import pandas as pd\nimport matplotlib.pyplot as plt\n"
            primer_code = primer_code + "fig,ax = plt.subplots(1,1)\n"
            primer_code = primer_code + "ax.spines['top'].set_visible(False)\nax.spines['right'].set_visible(False) \n"
            answer = primer_code + extracted_code
            plot_area.pyplot(exec(answer))
        except Exception as e:  # Response is not code
            print(f"Error executing response: {e}")
            print(answer)
            st.write("The bot did not return any Python code. Either re-run if a visualization output is expected or see the answer above.")

with chat_n:
    st.header("Chat with the MER Narratives")

    with st.expander("Example queries", expanded=True):
        st.markdown(
            '''
            - What are some challenges (narratives) associated with the indicator TX_PVLS?
            - How many mentions of stockouts (narratives) are there over time?
            - Can you look at all the narratives for 2023, indicator bundle Treatment, and suggest 5-10 themes to group them into? 
            - For those narratives that mention COVID, what are some challenges expressed?
            - What external conflicts are mentioned in the narratives?
            '''
        )
    
    with st.expander("Narratives preview (first 5 rows)"):
        st.dataframe(df1.head(), use_container_width=True)  # Show data

    user_query2 = st.text_area("Ask a question about the MER narratives.")

    if st.button("Submit", key = 3) and len(user_query2) > 0:

        response2 = agent1.invoke(user_query2)
        st.write(response2["output"])
        st.write(response2)
