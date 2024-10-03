from flask import Flask, render_template, request
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__)

# Load Google API key
google_api_key = 'AIzaSyAEAh9iTSnHtr6YOwrxOTWuGzpdJ5LXnug'

# Initialize the LangChain-based model gemini-1.5-pro-latest
llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest', temperature=0.7, google_api_key=google_api_key)

# Set up memory and conversation chain
memory = ConversationBufferMemory()
conversation_chain = ConversationChain(llm=llm, memory=memory)

# Function to generate resume summary
def generate_summary(resume_data):
    resume_summary_prompt = (
        f"Please generate a summary for the following resume:\n"
        f"Name: {resume_data['name']}\n"
        f"Summary: {resume_data['summary']}\n"
        f"University: {resume_data['university']}\n"
        f"Degree: {resume_data['degree']}\n"
        f"Graduation Year: {resume_data['graduation_year']}\n"
        f"Skills: {resume_data['skills']}\n"
        f"Languages: {resume_data['languages']}\n"
        f"Experience: {resume_data['experience']}\n"
        f"Address: {resume_data['address']}\n"
        f"Phone: {resume_data['phone']}\n"
        f"Email: {resume_data['email']}\n"
    )
    return conversation_chain.run(resume_summary_prompt)

# Route for displaying the resume form
@app.route('/')
def resume_form():
    return render_template('form.html')  # Display the form.html

# Route for handling the form submission
@app.route('/generate_resume', methods=['POST'])
def generate_resume():
    # Collect the form data
    resume_data = {
        'name': request.form['name'],
        'summary': request.form['summary'],
        'university': request.form['university'],
        'degree': request.form['degree'],
        'graduation_year': request.form['graduation_year'],
        'skills': request.form['skills'],
        'languages': request.form['languages'],
        'experience': request.form['experience'],
        'address': request.form['address'],
        'phone': request.form['phone'],
        'email': request.form['email']
    }

    # Generate resume summary using LangChain
    generated_summary = generate_summary(resume_data)

    # Render the result2.html template and pass the data along with the generated summary
    return render_template('result_new.html', 
                           resume=resume_data, 
                           generated_summary=generated_summary)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
