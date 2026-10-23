"""Regenerate the sample resumes shipped with the web UI (src/hrscreen/web/samples/resumes/).

All people, companies and contact details are fictional. The files exist so the upload forms can be
tried with one click; each is written in the format named in its file name (PDF, DOCX or TXT), and
one PDF is deliberately blank to show how unreadable files are reported.

Needs PyMuPDF (pip install pymupdf) for the PDFs and python-docx for the DOCX files. The generated
files are committed, so you only need to run this when you change the text below.

    python scripts/build_sample_resumes.py
"""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
from docx import Document

OUT = Path(__file__).resolve().parent.parent / "src" / "hrscreen" / "web" / "samples" / "resumes"
FOOTER = "(Sample resume: a fictional person, made up to test the screening tool.)"

# "# " starts a section heading and "- " a bullet; the first line is the person's name.
RESUMES = {
    "priya_nair_data_scientist.pdf": """Priya Nair
Data Scientist | priya.nair@example.com | +91 90000 00001
# Summary
Data scientist with 6 years of experience building predictive models for retail and logistics. Strong in Python, statistics and explaining results to business teams.
# Skills
Python, pandas, numpy, scikit-learn, xgboost, tensorflow, machine learning, deep learning, statistics, hypothesis testing, A/B testing, feature engineering, SQL, Spark, Tableau, Power BI, NLP, time series
# Experience
Senior Data Scientist, Northwind Analytics (2021 - present)
- Built demand forecasting models with time series methods and xgboost.
- Designed and analysed A/B tests and presented findings with Tableau dashboards.
- Mentored two junior analysts on feature engineering and model deployment.
Data Scientist, Orchid Labs (2018 - 2021)
- Trained NLP classifiers with scikit-learn and tensorflow for support tickets.
- Cleaned and analysed large datasets with pandas and SQL.
# Education
M.Tech in Statistics, 2018
B.Sc in Mathematics, 2016""",
    "rohan_mehta_python_developer.docx": """Rohan Mehta
Python Developer | rohan.mehta@example.com | +91 90000 00002
# Summary
Backend developer with 4 years of experience building REST APIs and services in Python.
# Skills
Python, Django, Flask, FastAPI, REST API, PostgreSQL, Redis, Celery, Docker, Git, pytest, unit testing, AWS, Linux, microservices, object oriented programming
# Experience
Python Developer, Bluepeak Systems (2022 - present)
- Built and maintained REST APIs with Django and FastAPI serving 200k requests a day.
- Moved background jobs to Celery and Redis, and added pytest coverage for the payment service.
- Containerised services with Docker and deployed them to AWS.
Junior Developer, Sunrise Software (2020 - 2022)
- Developed Flask services backed by PostgreSQL and wrote unit testing suites.
# Education
B.Tech in Computer Science, 2020""",
    "ananya_iyer_java_developer.txt": """Ananya Iyer
Java Developer | ananya.iyer@example.com | +91 90000 00003
# Summary
Java developer with 7 years of experience in enterprise applications and microservices.
# Skills
Java, Spring Boot, Spring Framework, Hibernate, JDBC, microservices, REST API, Kafka, MySQL, Oracle, JUnit, Maven, Jenkins, Git, multithreading, design patterns
# Experience
Senior Java Developer, Harbor Financial Systems (2020 - present)
- Designed Spring Boot microservices for loan processing and exposed REST APIs.
- Introduced Kafka for event streaming and improved throughput with multithreading.
- Wrote JUnit tests and automated builds with Maven and Jenkins.
Java Developer, Lotus Infotech (2017 - 2020)
- Built Hibernate based data layers on Oracle and MySQL.
# Education
B.Tech in Information Technology, 2017""",
    "karan_malhotra_devops_engineer.pdf": """Karan Malhotra
DevOps Engineer | karan.malhotra@example.com | +91 90000 00004
# Summary
DevOps engineer with 5 years of experience automating builds, releases and cloud infrastructure.
# Skills
Docker, Kubernetes, Terraform, Ansible, Jenkins, CI/CD, AWS, Azure, Linux, Bash, Python, Prometheus, Grafana, Nginx, infrastructure as code, monitoring
# Experience
DevOps Engineer, Cloudgrove Technologies (2021 - present)
- Built CI/CD pipelines with Jenkins and moved services to Kubernetes on AWS.
- Managed infrastructure with Terraform and Ansible.
- Set up Prometheus and Grafana monitoring and alerting.
Systems Engineer, Delta Hosting (2019 - 2021)
- Administered Linux servers and Nginx, scripting routine tasks in Bash and Python.
# Education
B.Sc in Information Technology, 2019""",
    "sneha_kulkarni_web_designer.txt": """Sneha Kulkarni
Web Designer | sneha.kulkarni@example.com | +91 90000 00005
# Summary
Web designer with 3 years of experience creating responsive, accessible websites for small businesses.
# Skills
HTML5, CSS3, JavaScript, React, Bootstrap, Figma, Adobe XD, Photoshop, Illustrator, responsive design, UI design, UX design, WordPress, SEO, wireframing
# Experience
Web Designer, Pixel Harbour Studio (2022 - present)
- Designed wireframes and prototypes in Figma and Adobe XD and built them with HTML5, CSS3 and React.
- Made sites responsive with Bootstrap and tested them across browsers.
- Improved page speed and basic SEO for WordPress clients.
Design Intern, Brightside Media (2021 - 2022)
- Produced banners and layouts in Photoshop and Illustrator.
# Education
Diploma in Graphic Design, 2021""",
    "farah_khan_hr_executive.docx": """Farah Khan
HR Executive | farah.khan@example.com | +91 90000 00006
# Summary
HR executive with 5 years of experience across recruitment, onboarding and employee engagement.
# Skills
Recruitment, talent acquisition, sourcing, interviewing, onboarding, induction, payroll, employee engagement, performance management, HRIS, labour law, statutory compliance, grievance handling, HR analytics
# Experience
HR Executive, Greenfield Manufacturing (2021 - present)
- Ran end to end recruitment for 60 hires a year, from sourcing to onboarding and induction.
- Coordinated monthly payroll and statutory compliance.
- Tracked attrition in the HRIS and reported HR analytics to management.
HR Associate, Meridian Services (2019 - 2021)
- Supported performance management cycles and handled employee grievances.
# Education
MBA in Human Resources, 2019""",
    "vikram_singh_sales_executive.txt": """Vikram Singh
Sales Executive | vikram.singh@example.com | +91 90000 00007
# Summary
Sales executive with 6 years of experience in B2B sales and key account management.
# Skills
Lead generation, cold calling, CRM, Salesforce, negotiation, B2B, account management, key account management, sales forecasting, pipeline management, customer retention, market research, presentations
# Experience
Senior Sales Executive, Summit Industrial Supplies (2020 - present)
- Grew revenue from key accounts by managing the pipeline in Salesforce and negotiating annual contracts.
- Met or beat the monthly target in 31 of the last 36 months.
Sales Executive, Coastline Distributors (2018 - 2020)
- Generated leads through cold calling and market research, then closed B2B deals.
# Education
B.Com, 2018""",
    "deepa_reddy_business_analyst.docx": """Deepa Reddy
Business Analyst | deepa.reddy@example.com | +91 90000 00008
# Summary
Business analyst with 5 years of experience turning stakeholder needs into clear requirements.
# Skills
Requirements gathering, business analysis, BRD, FRD, user stories, use cases, process mapping, gap analysis, stakeholder management, Jira, Confluence, SQL, Agile, Scrum, UAT, Power BI, Excel
# Experience
Business Analyst, Crestline Insurance (2021 - present)
- Ran requirements gathering workshops and wrote BRD, FRD and user stories in Jira and Confluence.
- Analysed claims data with SQL and Power BI and supported UAT in Agile and Scrum teams.
Junior Analyst, Opal Consulting (2019 - 2021)
- Mapped processes and performed gap analysis for client projects.
# Education
MBA in Operations, 2019""",
    "arjun_das_civil_engineer.txt": """Arjun Das
Civil Engineer | arjun.das@example.com | +91 90000 00009
# Summary
Civil engineer with 8 years of experience on residential and commercial building projects.
# Skills
AutoCAD, Revit, STAAD Pro, structural analysis, RCC design, site supervision, quantity surveying, estimation and costing, BOQ, billing, Primavera, quality control, project planning, concrete
# Experience
Site Engineer, Keystone Constructions (2019 - present)
- Supervised site execution and quality control for a 12 storey building.
- Prepared BOQ, estimation and costing, and monthly billing.
- Planned the schedule in Primavera.
Junior Engineer, Granite Builders (2016 - 2019)
- Prepared drawings in AutoCAD and checked RCC designs with STAAD Pro.
# Education
B.Tech in Civil Engineering, 2016""",
}


def wrap(text: str, width: float, font: str, size: float) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if fitz.get_text_length(trial, fontname=font, fontsize=size) <= width:
            current = trial
        else:
            lines.append(current)
            current = word
    return lines + [current]


def write_pdf(path: Path, text: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    y, left, width = 60, 56, page.rect.width - 112
    for n, raw in enumerate(text.splitlines() + ["", FOOTER]):
        heading = raw.startswith("# ")
        body = raw[2:] if heading else raw
        font, size = ("hebo", 12) if (heading or n == 0) else ("helv", 10)
        if n == 0:
            size = 16
        for line in wrap(body, width, font, size) if body else [""]:
            page.insert_text((left, y), line, fontname=font, fontsize=size)
            y += size + 4
        y += 4 if heading else 0
        if y > page.rect.height - 50:
            raise SystemExit(f"{path.name} does not fit on one page")
    doc.save(path)
    doc.close()


def write_docx(path: Path, text: str) -> None:
    doc = Document()
    for n, raw in enumerate(text.splitlines() + ["", FOOTER]):
        if n == 0:
            doc.add_heading(raw, level=1)
        elif raw.startswith("# "):
            doc.add_heading(raw[2:], level=2)
        elif raw.startswith("- "):
            doc.add_paragraph(raw[2:], style="List Bullet")
        elif raw:
            doc.add_paragraph(raw)
    doc.save(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.iterdir():
        old.unlink()
    for name, text in RESUMES.items():
        path = OUT / name
        if name.endswith(".pdf"):
            write_pdf(path, text)
        elif name.endswith(".docx"):
            write_docx(path, text)
        else:
            path.write_text(text + "\n\n" + FOOTER + "\n", encoding="utf-8")
    blank = fitz.open()
    blank.new_page()
    blank.save(OUT / "blank_scan_unreadable.pdf")   # no text layer: shows the "No text found" message
    blank.close()
    print(f"wrote {len(list(OUT.iterdir()))} files to {OUT}")


if __name__ == "__main__":
    main()
