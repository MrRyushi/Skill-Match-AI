from huggingface_hub import InferenceClient
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv

# load the pdf
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def send_to_huggingface(resume_text, job_description):
    client = InferenceClient()
    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3-0324",
        messages=[
            {
                "role": "user",
                "content": f"""
                You are an AI Resume Analyzer. 
                Input: 
                1. Resume text: {resume_text}
                2. Job description text {job_description}

                Task:
                - Extract main skills from the resume
                - Extract required skills from the job description
                - Compare and calculate a match score from 0 to 100
                - List matched skills, missing skills, and recommendations for the candidate to improve
                Return output in JSON format:
                {{ 
                "match_score": int,
                "resume_skills": [..],
                "job_skills": [..],
                "matched_skills": [..],
                "missing_skills": [..],
                "recommendations": [..]
                }}
                    
                
                """
            }
        ],
    )
    print(completion.choices[0].message.content)

# get the key



resume_text = load_pdf("resume.pdf")
job_description = f"""
Work setup: Hybrid (open to 2x a week in the office)

Work schedule: 10AM to 6PM Manila time 

Employment type: Permanent

Location: Makati City, Metro Manila

Pay range: Php 60,000 to Php 81,000

We value transparency and encourage applicants comfortable with this range to apply.

 

Discover a world of endless possibilities with Cambridge University Press & Assessment, a distinguished global academic publisher and assessment organisation proudly affiliated with the prestigious University of Cambridge.

 

We are looking for a Software Engineer to join our Manila Academic Development Team. As a key member of our team, your goal is to develop and maintain JS stack web applications of Academic Technology that publishes journals, articles and books for Cambridge University and its affiliates across the world. This role will play an essential part in the support and development of Cambridge Core and its key content and platform applications.

 

Why Cambridge?

Cambridge University Press & Assessment is a world-renowned not-for-profit academic publisher and assessment organisation, proudly part of the prestigious University of Cambridge. With a legacy rooted in over 800 years of educational excellence, we are dedicated to unlocking the potential of learners and educators across the globe.

Joining Cambridge's second largest global office in the Philippines —operating for over 22 years with 1,300+ colleagues— means becoming a part of an extraordinary institution renowned worldwide. We are recognised as a Great Place to Work® for three consecutive years, reflecting our inclusive culture, strong sense of purpose, and commitment to the professional growth and well-being of our people. At Cambridge, we don't just publish books or deliver tests—we empower progress, inspire curiosity, and champion the pursuit of knowledge.

 

What can you get from Cambridge?

At Cambridge, you'll become a part of a vibrant and forward-thinking community that transcends tradition, fostering a culture of continuous growth and personal development. Here, we provide the right environment for you to thrive, supporting your professional journey and empowering you to reach your highest potential, that is why our pay philosophy is intricately tied to your skills and competencies, ensuring that your compensation aligns with the unique value you bring to the role you are applying for.

The organization offers a wide range of benefits and opportunities including:

Regular Employment on Day 1
HMO Coverage and Life Insurance on Day 1
Paid Annual Leaves (Vacation, Well-being, Flexible, Holiday, and Volunteering leaves)
Vesting/Retirement package
Opportunities for career growth and development
Access to well-being programs
Flexible schedule, hybrid work arrangement and work-life balance
Opportunity to collaborate with colleagues from diverse branches that will expand your horizons and enrich your understanding of different cultures
 

What will you do as a Software Engineer?

 As a Software Engineer you will play a crucial role in developing and supporting web applications for Cambridge Core and other applications under the Academic Development Team's responsibility. You will work closely with other software developers using primarily JS tech stack and the latest JS frameworks.

 

Please review the attached job description for further details on the role. 

 

What makes you the ideal candidate for this role?

Full Stack JavaScript Development: 1–2 years of experience working in a full stack environment, including Node.js, Express.js, Handlebars.js, Vue.js, and Nuxt.js.

Frontend Development: Over 2 years of hands-on experience in building responsive web applications using HTML, CSS, and JavaScript, with additional exposure to Foundation5 as a frontend framework.

 

Are you driven by desire to be part of a globally renowned institution that celebrates innovation, embraces inclusion, and empowers learners? Then, we invite you to Pursue your Potential with us.

 

Applications received through the system will be reviewed on a rolling basis and may close the vacancy once sufficient applications are received. Therefore, if you are interested, tailor-fit your CV (advantageous if you submit one with a Cover Letter) and submit as early as possible.

"""
send_to_huggingface(resume_text, job_description)
