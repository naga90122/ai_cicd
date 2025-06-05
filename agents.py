# agents.py - Multi-Agent System Implementation
import os
import json
import time
import base64
import requests
import subprocess
import chromadb
import fitz  # PyMuPDF
from github import Github, GithubException
from duckduckgo_search import DDGS
from chromadb.utils import embedding_functions
from typing import Dict, List, Optional, Tuple

# Configuration
class Config:
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
    GITHUB_REPO = "naga90122/ai_cicd"
    VECTORDB_PATH = "./vector_db"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_API_URL = "http://localhost:5000/v1/completions"  # Local LLM endpoint

# Base Agent Class
class Agent:
    def __init__(self, name: str):
        self.name = name
        self.log_history = []
    
    def log(self, message: str):
        entry = f"[{time.ctime()}] {self.name}: {message}"
        self.log_history.append(entry)
        print(entry)
    
    def clear_logs(self):
        self.log_history = []
    
    def query_llm(self, prompt: str, max_tokens=512) -> str:
        """Query local LLM API"""
        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            response = requests.post(Config.LLM_API_URL, json=data, headers=headers)
            return response.json()["choices"][0]["text"].strip()
        except Exception as e:
            self.log(f"LLM Error: {str(e)}")
            return ""

# GitHub Agent
class GitHubAgent(Agent):
    def __init__(self):
        super().__init__("GitHubAgent")
        if not Config.GITHUB_TOKEN:
            raise ValueError("Missing GITHUB_TOKEN in environment variables")
        
        self.auth = Github(Config.GITHUB_TOKEN)
        try:
            self.repo = self.auth.get_repo(Config.GITHUB_REPO)
        except GithubException:
            self.repo = self.create_repo(Config.GITHUB_REPO)
    
    def create_repo(self, repo_name: str) -> str:
        """Create new repository if not exists"""
        user = self.auth.get_user()
        repo = user.create_repo(repo_name, auto_init=True)
        self.log(f"Created new repository: {repo_name}")
        return repo
    
    def commit_files(self, files: Dict[str, str], commit_message: str):
        """Commit multiple files to repository"""
        try:
            # Get current commit SHA
            branch = self.repo.get_branch("main")
            base_tree = self.repo.get_git_tree(sha=branch.commit.sha)
            
            # Create blobs
            blobs = []
            for path, content in files.items():
                if isinstance(content, str):
                    content = content.encode("utf-8")
                blob = self.repo.create_git_blob(content.decode("utf-8") if isinstance(content, bytes) else content, "utf-8")
                blobs.append({
                    "path": path,
                    "mode": "100644",
                    "type": "blob",
                    "sha": blob.sha
                })
            
            # Create new tree
            new_tree = self.repo.create_git_tree(blobs, base_tree)
            new_commit = self.repo.create_git_commit(
                message=commit_message,
                tree=new_tree,
                parents=[branch.commit.sha]
            )
            
            # Update reference
            self.repo.get_git_ref("heads/main").edit(new_commit.sha)
            self.log(f"Committed {len(files)} files: {commit_message}")
            return True
        except GithubException as e:
            self.log(f"Commit failed: {str(e)}")
            return False
    
    def push_local_repo(self, local_path: str):
        """Push local repository changes"""
        try:
            # Configure remote URL with token
            subprocess.run(
                ["git", "remote", "set-url", "origin", 
                 f"https://{Config.GITHUB_TOKEN}@github.com/{Config.GITHUB_REPO}.git"],
                cwd=local_path,
                check=True
            )
            
            # Push changes
            subprocess.run(
                ["git", "push", "origin", "main"],
                cwd=local_path,
                check=True
            )
            self.log("Pushed local repository to GitHub")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"Git push failed: {str(e)}")
            return False

# VectorDB Agent
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
            
            self.collection.add(
                documents=[text],
                metadatas=[metadata] if metadata else [{}],
                ids=[file_path]
            )
            self.log(f"Stored PDF: {file_path} ({len(text)} characters)")
            return True
        except Exception as e:
            self.log(f"PDF storage failed: {str(e)}")
            return False
    
    def store_text(self, text: str, doc_id: str, metadata: Optional[dict] = None):
        """Store raw text in VectorDB"""
        self.collection.add(
            documents=[text],
            metadatas=[metadata] if metadata else [{}],
            ids=[doc_id]
        )
        self.log(f"Stored text document: {doc_id}")
    
    def query(self, query_text: str, n_results: int = 3) -> List[Tuple[str, dict]]:
        """Query VectorDB for similar documents"""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        return list(zip(docs, metadatas))

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
            self.log(f"Search failed: {str(e)}")
            return []

# Architect Agent
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
        except json.JSONDecodeError:
            self.log("Failed to parse design as JSON")
            return {
                "overview": "Design generation failed",
                "components": [],
                "tech_stack": [],
                "diagram_description": ""
            }

# Developer Agent
class DeveloperAgent(Agent):
    def __init__(self):
        super().__init__("DeveloperAgent")
    
    def write_code(self, specification: str, language: str = "python") -> Dict[str, str]:
        """Generate code based on specification"""
        prompt = f"""You are a senior {language} developer. Write code based on this specification:
        
        {specification}
        
        Provide output in this JSON format:
        {{
            "files": {{
                "filename1.py": "code content",
                "filename2.js": "code content"
            }},
            "dependencies": ["list", "of", "packages"]
        }}
        """
        response = self.query_llm(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.log("Failed to parse code as JSON")
            return {"files": {}, "dependencies": []}

# QA Agent
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
                    "name": "Test case name",
                    "description": "Test description",
                    "steps": ["step1", "step2"],
                    "expected": "Expected result"
                }}
            ],
            "automated_tests": {{
                "test_filename.py": "test code content"
            }}
        }}
        """
        response = self.query_llm(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.log("Failed to parse tests as JSON")
            return {"test_cases": [], "automated_tests": {}}

# DevOps Agent
class DevOpsAgent(Agent):
    def __init__(self):
        super().__init__("DevOpsAgent")
    
    def create_dockerfile(self, dependencies: List[str]) -> str:
        """Generate Dockerfile based on dependencies"""
        prompt = f"""Create a Dockerfile for an application with these dependencies:
        
        {', '.join(dependencies)}
        
        Output ONLY the Dockerfile content without any additional text.
        """
        return self.query_llm(prompt)
    
    def create_ci_cd(self) -> str:
        """Generate CI/CD pipeline configuration"""
        prompt = """Create a GitHub Actions CI/CD pipeline configuration that:
        1. Runs tests on push
        2. Builds Docker image
        3. Deploys to Heroku
        
        Output ONLY the YAML content without any additional text.
        """
        return self.query_llm(prompt)

# Master Agent
class MasterAgent:
    def __init__(self):
        self.agents = {
            "github": GitHubAgent(),
            "vectordb": VectorDBAgent(),
            "search": SearchAgent(),
            "architect": ArchitectAgent(),
            "developer": DeveloperAgent(),
            "qa": QAAgent(),
            "devops": DevOpsAgent()
        }
        self.task_log = []
    
    def log_task(self, message: str):
        entry = f"[{time.ctime()}] Master: {message}"
        self.task_log.append(entry)
        print(entry)
    
    def execute_task(self, prompt: str):
        """Orchestrate agents to complete a task based on prompt"""
        self.log_task(f"Starting task: {prompt}")
        
        # Step 1: Research with SearchAgent
        self.log_task("Researching related information...")
        research = self.agents["search"].search_web(prompt, n_results=3)
        
        # Step 2: Design with ArchitectAgent
        self.log_task("Creating system design...")
        design = self.agents["architect"].design_system(prompt)
        
        # Step 3: Develop with DeveloperAgent
        self.log_task("Generating code...")
        code_output = self.agents["developer"].write_code(
            f"Requirements: {prompt}\nDesign: {json.dumps(design, indent=2)}"
        )
        
        # Step 4: QA with QAAgent
        self.log_task("Creating tests...")
        tests = self.agents["qa"].test_code(code_output["files"])
        
        # Step 5: DevOps setup
        self.log_task("Generating deployment config...")
        dockerfile = self.agents["devops"].create_dockerfile(code_output.get("dependencies", []))
        ci_cd = self.agents["devops"].create_ci_cd()
        
        # Step 6: Prepare all files
        all_files = {}
        all_files.update(code_output["files"])
        all_files.update(tests["automated_tests"])
        all_files["Dockerfile"] = dockerfile
        all_files[".github/workflows/cicd.yaml"] = ci_cd
        all_files["DESIGN.md"] = json.dumps(design, indent=2)
        all_files["RESEARCH.md"] = "\n\n".join(
            [f"## {r['title']}\nURL: {r['url']}\n{r['snippet']}" for r in research]
        )
        
        # Step 7: Store in VectorDB
        self.log_task("Storing documentation in VectorDB...")
        self.agents["vectordb"].store_text(
            text=json.dumps({
                "prompt": prompt,
                "design": design,
                "research": research
            }),
            doc_id=f"task_{int(time.time())}"
        )
        
        # Step 8: Commit to GitHub
        self.log_task("Committing to GitHub...")
        commit_result = self.agents["github"].commit_files(
            files=all_files,
            commit_message=f"AI-generated solution for: {prompt[:50]}..."
        )
        
        self.log_task(f"Task completed: {'Success' if commit_result else 'Failed'}")

# Example Usage
if __name__ == "__main__":
    # Initialize master agent
    master = MasterAgent()
    
    # Example task
    task_prompt = "Create a weather dashboard using React that shows 7-day forecast"
    master.execute_task(task_prompt)
    
    print("\nTask Log:")
    for entry in master.task_log:
        print(entry)
