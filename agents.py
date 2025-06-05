%%writefile agents.py
# agents.py - Enhanced Kaggle-Compatible Multi-Agent System
import os
import json
import time
import base64
import requests
import chromadb
import fitz  # PyMuPDF
import traceback
import logging
from datetime import datetime
from duckduckgo_search import DDGS
from chromadb.utils import embedding_functions
from typing import Dict, List, Optional, Tuple, Any

# Centralized Logging Setup
class LoggerAgent:
    _instance = None
    
    def __new__(cls, log_file="/kaggle/working/agent_system.log"):
        if cls._instance is None:
            cls._instance = super(LoggerAgent, cls).__new__(cls)
            cls._instance.setup_logger(log_file)
        return cls._instance
    
    def setup_logger(self, log_file):
        self.logger = logging.getLogger("AgentSystem")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log(self, agent_name: str, message: str, level: str = "info"):
        log_message = f"[{agent_name}] {message}"
        if level == "info":
            self.logger.info(log_message)
        elif level == "error":
            self.logger.error(log_message)
        elif level == "warning":
            self.logger.warning(log_message)
        print(log_message)

# Configuration for Kaggle
class Config:
    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
    GITHUB_REPO = "naga90122/ai_cicd"
    VECTORDB_PATH = "/kaggle/working/vector_db"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

# GitHub API Implementation
class GitHubAPI:
    BASE_URL = "https://api.github.com"
    
    def __init__(self, token):
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json"
        }
    
    def get_repo(self, repo_name):
        url = f"{self.BASE_URL}/repos/{repo_name}"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return self.create_repo(repo_name.split("/")[1])
        else:
            raise Exception(f"GitHub API Error: {response.status_code} - {response.text}")
    
    def create_repo(self, repo_name, private=True):
        url = f"{self.BASE_URL}/user/repos"
        data = {
            "name": repo_name,
            "private": private,
            "auto_init": True
        }
        response = requests.post(url, json=data, headers=self.headers)
        if response.status_code == 201:
            return response.json()
        else:
            raise Exception(f"Create repo failed: {response.status_code} - {response.text}")
    
    def create_file(self, repo_name, path, content, message, branch="main"):
        url = f"{self.BASE_URL}/repos/{repo_name}/contents/{path}"
        data = {
            "message": message,
            "content": base64.b64encode(content.encode()).decode(),
            "branch": branch
        }
        response = requests.put(url, json=data, headers=self.headers)
        if response.status_code in [200, 201]:
            return response.json()
        else:
            raise Exception(f"Create file failed: {response.status_code} - {response.text}")

# Base Agent Class with centralized logging
class Agent:
    def __init__(self, name: str):
        self.name = name
        self.logger = LoggerAgent()
    
    def log(self, message: str, level: str = "info"):
        self.logger.log(self.name, message, level)
    
    def query_llm(self, prompt: str, max_tokens=512) -> str:
        """Query Hugging Face LLM API with robust error handling"""
        try:
            headers = {
                "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
                "Content-Type": "application/json"
            }
            data = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": 0.7,
                    "return_full_text": False
                }
            }
            response = requests.post(Config.LLM_API_URL, json=data, headers=headers, timeout=120)
            
            if response.status_code != 200:
                self.log(f"LLM API Error: {response.status_code} - {response.text}", "error")
                return ""
            
            response_json = response.json()
            
            # Handle different response formats
            if isinstance(response_json, list):
                if 'generated_text' in response_json[0]:
                    return response_json[0]['generated_text'].strip()
                return response_json[0].get('text', '').strip()
            elif isinstance(response_json, dict):
                if 'error' in response_json:
                    self.log(f"LLM Error: {response_json['error']}", "error")
                return response_json.get('generated_text', '').strip()
            return ""
        except Exception as e:
            self.log(f"LLM Error: {str(e)}", "error")
            return ""

# GitHub Agent using direct API calls
class GitHubAgent(Agent):
    def __init__(self):
        super().__init__("GitHubAgent")
        if not Config.GITHUB_TOKEN:
            raise ValueError("GITHUB_TOKEN environment variable not set")
        
        self.api = GitHubAPI(Config.GITHUB_TOKEN)
        self.repo = self.api.get_repo(Config.GITHUB_REPO)
    
    def commit_files(self, files: Dict[str, str], commit_message: str):
        """Commit multiple files to repository"""
        try:
            for path, content in files.items():
                self.api.create_file(
                    repo_name=Config.GITHUB_REPO,
                    path=path,
                    content=content,
                    message=commit_message
                )
            self.log(f"Committed {len(files)} files: {commit_message}")
            return True
        except Exception as e:
            self.log(f"Commit failed: {str(e)}", "error")
            return False

# Fixed VectorDB Agent
class VectorDBAgent(Agent):
    def __init__(self):
        super().__init__("VectorDBAgent")
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=Config.EMBEDDING_MODEL
        )
        self.client = chromadb.PersistentClient(path=Config.VECTORDB_PATH)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_fn
        )
    
    def store_pdf(self, file_path: str, metadata: Optional[dict] = None) -> bool:
        """Extract text from PDF and store in VectorDB"""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            
            # Fixed metadata handling
            self.collection.add(
                documents=[text],
                metadatas=[metadata] if metadata else None,
                ids=[file_path]
            )
            self.log(f"Stored PDF: {file_path} ({len(text)} characters)")
            return True
        except Exception as e:
            self.log(f"PDF storage failed: {str(e)}", "error")
            return False
    
    def store_text(self, text: str, doc_id: str, metadata: Optional[dict] = None):
        """Store raw text in VectorDB with fixed metadata"""
        try:
            self.collection.add(
                documents=[text],
                metadatas=[metadata] if metadata else None,
                ids=[doc_id]
            )
            self.log(f"Stored text document: {doc_id}")
            return True
        except Exception as e:
            self.log(f"VectorDB storage failed: {str(e)}", "error")
            return False

# Search Agent
class SearchAgent(Agent):
    def __init__(self):
        super().__init__("SearchAgent")
    
    def search_web(self, query: str, n_results: int = 5) -> List[Dict[str, str]]:
        """Search the web using DuckDuckGo"""
        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=n_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")
                    })
            self.log(f"Found {len(results)} results for: {query}")
            return results
        except Exception as e:
            self.log(f"Search failed: {str(e)}", "error")
            return []

# Architect Agent with JSON fallback
class ArchitectAgent(Agent):
    def __init__(self):
        super().__init__("ArchitectAgent")
    
    def design_system(self, requirements: str) -> Dict[str, str]:
        """Generate system design based on requirements"""
        prompt = f"""You are a solutions architect. Design a system based on these requirements:
        
        {requirements}
        
        Provide output in this JSON format:
        {{
            "overview": "System overview",
            "components": ["list", "of", "components"],
            "tech_stack": ["list", "of", "technologies"],
            "diagram_description": "Mermaid.js diagram description"
        }}
        """
        response = self.query_llm(prompt)
        try:
            return json.loads(response)
        except:
            # Fallback to simple design if JSON parsing fails
            self.log("Using fallback design", "warning")
            return {
                "overview": "Weather dashboard with React frontend",
                "components": ["Frontend UI", "Weather API client", "Data storage"],
                "tech_stack": ["React", "Node.js", "Express"],
                "diagram_description": "flowchart LR\nA[React Frontend] --> B[Weather API]\nB --> C[Data Cache]"
            }

# Developer Agent with JSON fallback
class DeveloperAgent(Agent):
    def __init__(self):
        super().__init__("DeveloperAgent")
    
    def write_code(self, specification: str) -> Dict[str, str]:
        """Generate code based on specification"""
        prompt = f"""You are a senior developer. Write code based on this specification:
        
        {specification}
        
        Provide output in this JSON format:
        {{
            "files": {{
                "src/App.js": "React code here",
                "src/components/WeatherCard.js": "Component code here"
            }},
            "dependencies": ["react", "axios"]
        }}
        """
        response = self.query_llm(prompt)
        try:
            return json.loads(response)
        except:
            # Fallback to simple code
            self.log("Using fallback code", "warning")
            return {
                "files": {
                    "src/App.js": "import React from 'react';\n\nfunction App() {\n  return <div>Weather Dashboard</div>;\n}\n\nexport default App;",
                    "src/components/WeatherCard.js": "import React from 'react';\n\nconst WeatherCard = () => {\n  return <div>Weather Card</div>;\n};\n\nexport default WeatherCard;"
                },
                "dependencies": ["react", "react-dom"]
            }

# QA Agent with JSON fallback
class QAAgent(Agent):
    def __init__(self):
        super().__init__("QAAgent")
    
    def test_code(self, code: Dict[str, str]) -> Dict[str, str]:
        """Generate test cases for provided code"""
        prompt = f"""You are a QA engineer. Create test cases for this code:
        
        {json.dumps(code, indent=2)}
        
        Provide output in this JSON format:
        {{
            "test_cases": [
                {{
                    "name": "Render test",
                    "description": "Component renders without crashing",
                    "steps": ["Import component", "Render in test environment"],
                    "expected": "Component renders successfully"
                }}
            ],
            "automated_tests": {{
                "src/App.test.js": "Test code here"
            }}
        }}
        """
        response = self.query_llm(prompt)
        try:
            return json.loads(response)
        except:
            # Fallback to simple tests
            self.log("Using fallback tests", "warning")
            return {
                "test_cases": [{
                    "name": "Basic render",
                    "description": "Main app renders without crashing",
                    "steps": ["Load App component", "Render in DOM"],
                    "expected": "No errors in console"
                }],
                "automated_tests": {
                    "src/App.test.js": "import React from 'react';\nimport { render } from '@testing-library/react';\nimport App from './App';\n\ntest('renders without crashing', () => {\n  render(<App />);\n});"
                }
            }

# DevOps Agent
class DevOpsAgent(Agent):
    def __init__(self):
        super().__init__("DevOpsAgent")
    
    def create_dockerfile(self, dependencies: List[str]) -> str:
        """Generate Dockerfile based on dependencies"""
        prompt = f"""Create a Dockerfile for a React application with these dependencies:
        
        {', '.join(dependencies)}
        
        Output ONLY the Dockerfile content without any additional text.
        """
        response = self.query_llm(prompt)
        if not response:
            # Fallback Dockerfile
            return """FROM node:18
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]"""
        return response
    
    def create_ci_cd(self) -> str:
        """Generate CI/CD pipeline configuration"""
        prompt = """Create a GitHub Actions CI/CD pipeline configuration that:
        1. Runs tests on push
        2. Builds Docker image
        3. Deploys to Heroku
        
        Output ONLY the YAML content without any additional text.
        """
        response = self.query_llm(prompt)
        if not response:
            # Fallback CI/CD
            return """name: CI/CD Pipeline
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: 18
      - run: npm install
      - run: npm test
      - name: Build Docker image
        run: docker build -t weather-app .
      - name: Deploy to Heroku
        uses: akhileshns/heroku-deploy@v3.12.12
        with:
          heroku_api_key: ${{ secrets.HEROKU_API_KEY }}
          heroku_app_name: "your-app-name"
          heroku_email: "your-email@example.com\""""
        return response

# Troubleshooting Agent
class TroubleshootingAgent(Agent):
    def __init__(self):
        super().__init__("TroubleshootingAgent")
    
    def analyze_error(self, error: Exception, context: str, task_data: Dict[str, Any]) -> str:
        """Analyze errors and suggest solutions"""
        error_trace = traceback.format_exc()
        error_msg = str(error)
        
        prompt = f"""You are a troubleshooting expert. Analyze this error that occurred during task execution:
        
        Context: {context}
        Error: {error_msg}
        Traceback: {error_trace}
        Task Data: {json.dumps(task_data, indent=2)}
        
        Provide:
        1. Analysis of what went wrong
        2. Step-by-step solution to fix the issue
        3. Suggestions to prevent similar errors in the future
        """
        
        return self.query_llm(prompt)
    
    def validate_agent_output(self, agent_name: str, output: Any, expected_format: str) -> bool:
        """Validate agent output against expected format"""
        prompt = f"""Validate this output from {agent_name}:
        
        Output: {json.dumps(output, indent=2)}
        
        Expected Format: {expected_format}
        
        Report:
        1. Is the output valid?
        2. If not, what's missing or incorrect?
        3. How can it be fixed?
        """
        
        report = self.query_llm(prompt)
        self.log(f"Validation report for {agent_name}:\n{report}")
        return "valid: yes" in report.lower()

# Master Agent with Enhanced Error Handling
class MasterAgent:
    def __init__(self):
        self.logger = LoggerAgent()
        self.agents = {
            "github": GitHubAgent(),
            "vectordb": VectorDBAgent(),
            "search": SearchAgent(),
            "architect": ArchitectAgent(),
            "developer": DeveloperAgent(),
            "qa": QAAgent(),
            "devops": DevOpsAgent(),
            "troubleshooter": TroubleshootingAgent()
        }
        self.task_data = {}
    
    def log_task(self, message: str):
        self.logger.log("MasterAgent", message)
    
    def execute_step(self, step_name: str, step_func, *args, **kwargs):
        """Execute a step with error handling and validation"""
        try:
            self.log_task(f"Starting step: {step_name}")
            result = step_func(*args, **kwargs)
            self.log_task(f"Completed step: {step_name}")
            return result
        except Exception as e:
            error_msg = f"Step '{step_name}' failed: {str(e)}"
            self.log_task(error_msg)
            
            # Get troubleshooting report
            report = self.agents["troubleshooter"].analyze_error(
                e,
                step_name,
                {
                    "task_data": self.task_data,
                    "step": step_name,
                    "args": args,
                    "kwargs": kwargs
                }
            )
            
            self.log_task(f"Troubleshooting report:\n{report}")
            return None

    def execute_task(self, prompt: str):
        """Orchestrate agents to complete a task based on prompt"""
        self.log_task(f"Starting task: {prompt}")
        self.task_data = {"prompt": prompt, "start_time": datetime.now().isoformat()}
        
        # Step 1: Research with SearchAgent
        research = self.execute_step(
            "Research",
            self.agents["search"].search_web, 
            prompt, 
            n_results=3
        ) or []
        
        # Step 2: Design with ArchitectAgent
        design = self.execute_step(
            "Design System",
            self.agents["architect"].design_system,
            prompt
        ) or {
            "overview": "Fallback design due to errors",
            "components": [],
            "tech_stack": [],
            "diagram_description": ""
        }
        
        # Step 3: Develop with DeveloperAgent
        code_output = self.execute_step(
            "Code Generation",
            self.agents["developer"].write_code,
            f"Requirements: {prompt}\nDesign: {json.dumps(design, indent=2)}"
        ) or {
            "files": {},
            "dependencies": []
        }
        
        # Step 4: QA with QAAgent
        tests = self.execute_step(
            "Test Generation",
            self.agents["qa"].test_code,
            code_output.get("files", {})
        ) or {
            "test_cases": [],
            "automated_tests": {}
        }
        
        # Step 5: DevOps setup
        dockerfile = self.execute_step(
            "Dockerfile Creation",
            self.agents["devops"].create_dockerfile,
            code_output.get("dependencies", [])
        ) or "FROM node:18\nWORKDIR /app\nCOPY . .\nRUN npm install\nCMD ['npm', 'start']"
        
        ci_cd = self.execute_step(
            "CI/CD Creation",
            self.agents["devops"].create_ci_cd
        ) or "name: CI/CD Pipeline\non: [push]\njobs:\n  build:\n    runs-on: ubuntu-latest"
        
        # Step 6: Prepare all files
        self.log_task("Preparing files...")
        all_files = {}
        all_files.update(code_output.get("files", {}))
        all_files.update(tests.get("automated_tests", {}))
        all_files["Dockerfile"] = dockerfile
        all_files[".github/workflows/cicd.yml"] = ci_cd
        all_files["DESIGN.md"] = json.dumps(design, indent=2)
        all_files["RESEARCH.md"] = "\n\n".join(
            [f"## {r['title']}\nURL: {r['url']}\n{r['snippet']}" for r in research]
        ) if research else "# No research results"
        
        # Step 7: Store in VectorDB
        storage_result = self.execute_step(
            "VectorDB Storage",
            self.agents["vectordb"].store_text,
            text=json.dumps({
                "prompt": prompt,
                "design": design,
                "research": research,
                "code": list(code_output.get("files", {}).keys()),
                "tests": list(tests.get("automated_tests", {}).keys())
            }),
            doc_id=f"task_{int(time.time())}"
        )
        
        if not storage_result:
            self.log_task("VectorDB storage failed - continuing anyway")
        
        # Step 8: Commit to GitHub
        commit_result = self.execute_step(
            "GitHub Commit",
            self.agents["github"].commit_files,
            files=all_files,
            commit_message=f"AI-generated solution for: {prompt[:50]}..."
        )
        
        self.task_data["end_time"] = datetime.now().isoformat()
        self.task_data["status"] = "success" if commit_result else "partial_success"
        self.log_task(f"Task completed with status: {self.task_data['status']}")
        
        # Final validation
        self.log_task("Running final validation...")
        validation = self.agents["troubleshooter"].validate_agent_output(
            "MasterAgent",
            {
                "task_data": self.task_data,
                "output_files": list(all_files.keys()),
                "research_count": len(research),
                "code_files": len(code_output.get("files", {})),
                "test_cases": len(tests.get("test_cases", []))
            },
            "Expected: At least 2 code files and 1 test case"
        )

# Example Usage
if __name__ == "__main__":
    # Initialize master agent
    master = MasterAgent()
    
    # Example task
    task_prompt = "Create a weather dashboard using React that shows 7-day forecast"
    master.execute_task(task_prompt)
    
    print("\nTask completed. Check logs for details.")
