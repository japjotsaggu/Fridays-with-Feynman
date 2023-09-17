import os
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from IPython.display import display
import ipywidgets as widgets

def get_api_key():
	api_key = input("Enter your API key: ")
    return api_key

def pdf_loader(file):
	loader = PyPDFLoader(file)
	chunks = loader.load_and_split()
	return chunks

def vec_db(chunks):
	embeddings = HuggingFaceInstructEmbeddings(model_name = "hku-nlp/instructor-base")
	db = FAISS.from_documents(chunks, embeddings)
	return db

def main():
	chunks = PyPDFLoader("final_merging.pdf")
	#setting up the api environment 
	os.environ["OPENAI_API_KEY"] = get_api_key()  

	#Getting embedding model from HuggingFace and creating a vector database out of our chunks 
	db = vec_db(chunks)
	chain = load_qa_chain(OpenAI(temperature=0.1), chain_type="stuff")

	# Setting up chat history management - creating conversation chain that uses our db as retriver
	qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature = 0.1), db.as_retriever(), max_tokens_limit = 3000)

	#Bringing it all together - the actual widget 
	chat_history = []

	def on_submit(_):
	    query = input_box.value
	    input_box.value = ""
	    
	    if query.lower() == 'exit':
	        print("I enjoyed this chat, hope you did too! Come back soon to uncover more fascinating aspects of the world of Physics.")
	        return
	    
	    result = qa({"question": query, "chat_history": chat_history})
	    ans = result['answer']

	    prompt = f'''Explain the following in detail as Richard Feynman in first person:
	              "{ans}" '''

	    response = openai.Completion.create(engine="text-davinci-002",  prompt=prompt, max_tokens = 500, api_key = api_key)

	    final_ans = response.choices[0].text

	    chat_history.append((query, final_ans))
	    
	    display(widgets.HTML(f'<b>User:</b> {query}'))
	    display(widgets.HTML(f'<b><font color="blue">Mr.Feynman:</font></b> {final_ans}'))

	print("It's a fine day for some FeynPhysics! Type 'exit' to stop.")

	input_box = widgets.Text(placeholder='What fascinates you today?')
	input_box.on_submit(on_submit)

	display(input_box)

if __name__ == "__main__":
	main
