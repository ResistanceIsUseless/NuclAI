    def _create_bugbounty_examples(self, content: str, file_path: Path) -> List[Dict]:
        """Create training examples focused on bug bounty exploitation methodology"""
        examples = []
        
        # Extract vulnerability details and methodology
        vuln_details = self._extract_detailed_vulnerability_info(content)
        
        if not vuln_details:
            return examples
        
        # Example 1: Attack Methodology - "When you see X, try Y"
        if vuln_details.get('attack_vector'):
            examples.append({
                "messages": [
                    {"role": "user", "content": f"I found {vuln_details['indicators_found']} during reconnaissance. What attack should I try next?"},
                    {"role": "assistant", "content": f"Based on finding {vuln_details['indicators_found']}, this suggests a potential {vuln_details['vulnerability_type']} vulnerability. Here's the attack methodology:\\n\\n**Attack Vector:**\\n{vuln_details['attack_vector']}\\n\\n**Step-by-Step Exploitation:**\\n{vuln_details['exploitation_steps']}\\n\\n**Key Indicators to Confirm:**\\n{vuln_details['confirmation_indicators']}\\n\\n**Impact Assessment:**\\n{vuln_details['impact_description']}\\n\\n**Why This Works:**\\n{vuln_details['technical_explanation']}\\n\\nThis approach was successful in disclosed bug bounty reports because {vuln_details['success_factors']}."}
                ]
            })
        
        # Example 2: Exploitation Chain - Multiple steps
        if vuln_details.get('exploitation_chain'):
            examples.append({
                "messages": [
                    {"role": "user", "content": f"How do I escalate a {vuln_details['vulnerability_type']} finding into a more impactful vulnerability?"},
                    {"role": "assistant", "content": f"Here's how to escalate {vuln_details['vulnerability_type']} based on successful bug bounty techniques:\\n\\n**Escalation Chain:**\\n{vuln_details['exploitation_chain']}\\n\\n**Chaining Opportunities:**\\n{vuln_details['chaining_techniques']}\\n\\n**Impact Amplification:**\\n{vuln_details['impact_escalation']}\\n\\n**Real-world Example:**\\nIn disclosed reports, researchers combined {vuln_details['vulnerability_type']} with {vuln_details['secondary_vulns']} to achieve {vuln_details['final_impact']}.\\n\\n**Validation Steps:**\\n{vuln_details['validation_methods']}"}
                ]
            })
        
        # Example 3: Tool and Technique Selection
        if vuln_details.get('tools_used'):
            examples.append({
                "messages": [
                    {"role": "user", "content": f"What tools and techniques work best for finding and exploiting {vuln_details['vulnerability_type']}?"},
                    {"role": "assistant", "content": f"Based on successful bug bounty disclosures for {vuln_details['vulnerability_type']}, here are the most effective approaches:\\n\\n**Discovery Tools:**\\n{vuln_details['discovery_tools']}\\n\\n**Exploitation Techniques:**\\n{vuln_details['exploitation_techniques']}\\n\\n**Manual Testing Approach:**\\n{vuln_details['manual_testing_steps']}\\n\\n**Automation Strategy:**\\n{vuln_details['automation_tips']}\\n\\n**Pro Tips from Bug Bounty Hunters:**\\n{vuln_details['hunter_insights']}\\n\\n**Common Pitfalls to Avoid:**\\n{vuln_details['common_mistakes']}"}
                ]
            })
        
        # Example 4: Environmental Variations
        if vuln_details.get('environment_factors'):
            examples.append({
                "messages": [
                    {"role": "user", "content": f"I'm testing for {vuln_details['vulnerability_type']} but getting inconsistent results. What environmental factors should I consider?"},
                    {"role": "assistant", "content": f"Environmental factors significantly impact {vuln_details['vulnerability_type']} exploitation. Here's what to consider:\\n\\n**Technology Stack Variations:**\\n{vuln_details['technology_dependencies']}\\n\\n**Configuration Differences:**\\n{vuln_details['config_variations']}\\n\\n**Timing and Rate Limiting:**\\n{vuln_details['timing_considerations']}\\n\\n**Browser/Client Differences:**\\n{vuln_details['client_variations']}\\n\\n**Network Environment:**\\n{vuln_details['network_factors']}\\n\\n**Adaptation Strategies:**\\n{vuln_details['adaptation_techniques']}\\n\\nSuccessful bug bounty hunters adapt their approach based on these environmental cues to increase success rates."}
                ]
            })
        
        return examples
    
    def _extract_detailed_vulnerability_info(self, content: str) -> Dict:
        \"\"\"Extract detailed vulnerability information from bug bounty report\"\"\"
        vuln_info = {}
        
        # Identify vulnerability type
        vuln_type = self._identify_vulnerability_type(content)
        if not vuln_type:
            return {}
        
        vuln_info['vulnerability_type'] = vuln_type
        
        # Extract key information based on vulnerability type
        content_lower = content.lower()
        
        if vuln_type == 'SQL injection':
            vuln_info.update({
                'indicators_found': 'database error messages or unusual parameter behavior',
                'attack_vector': 'Parameter manipulation with SQL metacharacters',
                'exploitation_steps': '1. Identify injection points through error-based testing\\n2. Determine database type via fingerprinting\\n3. Extract database structure using UNION queries\\n4. Dump sensitive data or escalate to RCE',
                'confirmation_indicators': 'Time delays, boolean responses, or error messages revealing database structure',
                'impact_description': 'Data exfiltration, authentication bypass, potential RCE',
                'technical_explanation': 'Application fails to properly sanitize user input before including it in SQL queries',
                'success_factors': 'thorough parameter testing and creative payload encoding',
                'exploitation_chain': '1. Basic injection → 2. Database fingerprinting → 3. Privilege escalation → 4. File system access',
                'chaining_techniques': 'Combine with file upload for webshell deployment, or use for authentication bypass',
                'secondary_vulns': 'file upload vulnerabilities or weak access controls',
                'final_impact': 'complete application compromise and data breach',
                'validation_methods': 'Use time-based blind techniques and out-of-band data exfiltration',
                'discovery_tools': 'SQLMap, Burp Suite Active Scanner, manual parameter fuzzing',
                'exploitation_techniques': 'Union-based, error-based, boolean-based, and time-based blind injection',
                'manual_testing_steps': '1. Parameter enumeration\\n2. Error triggering\\n3. Payload crafting\\n4. Response analysis',
                'automation_tips': 'Use SQLMap with custom tamper scripts for WAF bypass',
                'hunter_insights': 'Focus on less obvious parameters like HTTP headers and JSON fields',
                'common_mistakes': 'Testing only GET parameters, ignoring JSON/XML inputs, not testing authenticated areas'
            })
        
        elif vuln_type == 'Cross-site scripting':
            vuln_info.update({
                'indicators_found': 'user input reflection or DOM manipulation capabilities',
                'attack_vector': 'JavaScript injection through input fields or URL parameters',
                'exploitation_steps': '1. Identify reflection points\\n2. Test basic XSS payloads\\n3. Bypass filters with encoding\\n4. Craft exploitation payload',
                'confirmation_indicators': 'JavaScript execution, alert boxes, or DOM modifications',
                'impact_description': 'Session hijacking, credential theft, defacement, or malware distribution',
                'technical_explanation': 'Application fails to properly encode user input before displaying it in HTML context',
                'success_factors': 'creative payload construction and filter bypass techniques',
                'exploitation_chain': '1. Reflected XSS → 2. Session theft → 3. Account takeover → 4. Privilege escalation',
                'chaining_techniques': 'Combine with CSRF for automated attacks, or use for phishing campaigns',
                'secondary_vulns': 'weak session management or insufficient access controls',
                'final_impact': 'complete account takeover and lateral movement',
                'validation_methods': 'Demonstrate data exfiltration and session hijacking',
                'discovery_tools': 'XSS Hunter, Burp Suite, manual payload testing',
                'exploitation_techniques': 'Reflected, stored, and DOM-based XSS with various encoding bypasses',
                'manual_testing_steps': '1. Input point mapping\\n2. Context analysis\\n3. Payload crafting\\n4. Filter bypass',
                'automation_tips': 'Use XSS Hunter for blind XSS detection and custom payloads for WAF bypass',
                'hunter_insights': 'Test in different contexts (HTML, JavaScript, CSS), focus on lesser-tested input fields',
                'common_mistakes': 'Only testing basic payloads, ignoring DOM XSS, not testing file upload functionality'
            })
        
        # Add more vulnerability types as needed
        elif vuln_type == 'Server-side request forgery':
            vuln_info.update({
                'indicators_found': 'URL parameters that fetch external resources or internal service calls',
                'attack_vector': 'Manipulating server to make requests to internal or external systems',
                'exploitation_steps': '1. Identify SSRF injection points\\n2. Test internal network access\\n3. Enumerate internal services\\n4. Exploit internal vulnerabilities',
                'confirmation_indicators': 'Response time differences, HTTP status variations, or content from internal services',
                'impact_description': 'Internal network reconnaissance, cloud metadata access, potential RCE',
                'technical_explanation': 'Server makes requests based on user input without proper validation of target URLs',
                'success_factors': 'understanding of internal network topology and cloud environments'
            })
        
        # Add common fields for all vulnerability types
        vuln_info.update({
            'technology_dependencies': 'Framework-specific behaviors, server configurations, and third-party integrations',
            'config_variations': 'WAF settings, input validation libraries, and security headers implementation',
            'timing_considerations': 'Rate limiting, request throttling, and session timeout behaviors',
            'client_variations': 'Browser-specific behaviors, JavaScript execution contexts, and mobile app differences',
            'network_factors': 'Load balancer configurations, CDN behaviors, and proxy server interactions',
            'adaptation_techniques': 'Payload encoding variations, timing adjustments, and alternative attack vectors'
        })
        
        return vuln_info    def generate_general_security_examples(self) -> List[Dict]:
        """Generate general security analysis and methodology examples"""
        examples = [
            {
                "messages": [
                    {"role": "user", "content": "I found several open ports during my network scan: 22, 80, 443, 3306, 5432. What should my next steps be?"},
                    {"role": "assistant", "content": "Great scan results! Here's a systematic approach for your next steps:\\n\\n**Immediate Priorities:**\\n1. **Port 3306 (MySQL)** - This is concerning if exposed externally. Test for default credentials, weak passwords, and version vulnerabilities\\n2. **Port 5432 (PostgreSQL)** - Similarly risky. Check authentication requirements and version\\n\\n**Standard Web Services:**\\n3. **Port 80/443 (HTTP/HTTPS)** - Run web application security scans, check for common vulnerabilities, analyze SSL configuration\\n4. **Port 22 (SSH)** - Test for weak credentials, check SSH version, verify key-based authentication\\n\\n**Systematic Approach:**\\n- Use service detection: `nmap -sV -sC` for detailed service info\\n- Run vulnerability scans: `nmap --script vuln` against each service\\n- Check for default credentials on database services\\n- Analyze web applications with tools like Nuclei or Burp Suite\\n\\nPrioritize the database ports as they pose the highest immediate risk if misconfigured."}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Create an Nmap NSE script to detect a custom service running on port 8080."},
                    {"role": "assistant", "content": "I'll create an NSE script for detecting your custom service on port 8080:\\n\\n```lua\\nlocal nmap = require \\\"nmap\\\"\\nlocal shortport = require \\\"shortport\\\"\\nlocal http = require \\\"http\\\"\\nlocal string = require \\\"string\\\"\\n\\ndescription = [[\\nDetects custom service running on port 8080 by analyzing response patterns.\\n]]\\n\\nauthor = \\\"Security Analyst\\\"\\nlicense = \\\"Same as Nmap--See https://nmap.org/book/man-legal.html\\\"\\ncategories = {\\\"discovery\\\", \\\"safe\\\"}\\n\\n-- Rule to run only on port 8080\\nportrule = shortport.port_or_service(8080, \\\"http\\\")\\n\\naction = function(host, port)\\n  local response = http.get(host, port, \\\"/\\\")\\n  \\n  if response and response.status == 200 then\\n    local body = response.body\\n    \\n    -- Check for custom service indicators\\n    if string.match(body, \\\"Custom%-Service\\\") or \\n       string.match(response.header[\\\"server\\\"] or \\\"\\\", \\\"CustomApp\\\") then\\n      \\n      return \\\"Custom service detected on port 8080\\\"\\n    end\\n  end\\n  \\n  return nil\\nend\\n```\\n\\nThis script:\\n- Targets port 8080 specifically\\n- Makes an HTTP GET request to the root path\\n- Checks response body and headers for custom service indicators\\n- Returns detection result\\n\\nModify the detection patterns to match your specific service's fingerprints."}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "How do I correlate findings from multiple security tools to reduce false positives?"},
                    {"role": "assistant", "content": "Correlating findings across multiple security tools is crucial for accurate assessment. Here's a systematic approach:\\n\\n**1. Tool Output Standardization:**\\n- Convert all findings to a common format (JSON/CSV)\\n- Normalize severity ratings across tools\\n- Map findings to standard vulnerability classifications (CWE/OWASP)\\n\\n**2. Correlation Strategies:**\\n- **IP/Port matching**: Group findings by target and service\\n- **Vulnerability type**: Correlate similar vulnerability classes\\n- **Confidence scoring**: Weight findings based on tool reliability\\n\\n**3. False Positive Reduction:**\\n- Require 2+ tools to confirm high-severity findings\\n- Manual verification of critical vulnerabilities\\n- Context analysis - consider business logic and environment\\n\\n**4. Practical Workflow:**\\n```\\nNmap (Discovery) → Nuclei (Vulnerability) → Burp (Manual) → Correlation\\n```\\n\\n**5. Automated Correlation:**\\n- Use tools like DefectDojo or Security Orchestration platforms\\n- Implement custom scripts to cross-reference findings\\n- Set up alert thresholds based on correlation confidence\\n\\n**Key Principle**: Higher confidence in findings that are detected by multiple independent tools using different methodologies."}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Walk me through creating a comprehensive security assessment methodology for a new web application."},
                    {"role": "assistant", "content": "Here's a comprehensive security assessment methodology for web applications:\\n\\n**Phase 1: Information Gathering (Passive)**\\n- **Domain reconnaissance**: Subdomain enumeration, DNS records\\n- **Technology stack identification**: Wappalyzer, BuiltWith, manual analysis\\n- **Public information**: Search engines, social media, code repositories\\n- **Certificate transparency logs**: crt.sh, Censys\\n\\n**Phase 2: Active Reconnaissance**\\n- **Port scanning**: Nmap comprehensive scan (`nmap -sS -sV -sC -O target`)\\n- **Web service discovery**: Directory/file enumeration with tools like ffuf, dirb\\n- **Technology fingerprinting**: Banner grabbing, error page analysis\\n- **SSL/TLS assessment**: SSLyze, testssl.sh\\n\\n**Phase 3: Vulnerability Assessment**\\n- **Automated scanning**: Nuclei templates, Burp Suite active scan\\n- **Web application testing**: OWASP Top 10 methodology\\n- **Authentication testing**: Login mechanisms, session management\\n- **Input validation**: SQL injection, XSS, command injection testing\\n\\n**Phase 4: Manual Testing**\\n- **Business logic flaws**: Application-specific vulnerabilities\\n- **Access control testing**: Horizontal/vertical privilege escalation\\n- **Session management**: Token analysis, logout functionality\\n- **File upload security**: Type validation, path traversal\\n\\n**Phase 5: Specialized Testing**\\n- **API security**: REST/GraphQL endpoint testing\\n- **Client-side security**: JavaScript analysis, DOM-based vulnerabilities\\n- **Infrastructure testing**: Server misconfigurations, container security\\n\\n**Phase 6: Validation and Documentation**\\n- **Manual verification**: Confirm all automated findings\\n- **Impact assessment**: Business impact analysis for each vulnerability\\n- **Proof of concept**: Develop exploits for critical findings\\n- **Remediation guidance**: Specific fix recommendations\\n\\n**Tools Integration Strategy:**\\n- **Discovery**: Nmap + Masscan + Amass\\n- **Vulnerability scanning**: Nuclei + Burp Suite + OWASP ZAP\\n- **Manual testing**: Burp Suite Professional + custom scripts\\n- **Reporting**: DefectDojo + custom reporting tools\\n\\n**Quality Assurance:**\\n- **Peer review**: Have findings validated by another analyst\\n- **Tool correlation**: Cross-reference findings between tools\\n- **False positive analysis**: Investigate and document false positives\\n- **Retest critical findings**: Verify exploitability in different scenarios\\n\\nThis methodology ensures comprehensive coverage while maintaining efficiency and accuracy in your security assessments."}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "What's the best approach for integrating security testing into a CI/CD pipeline?"},
                    {"role": "assistant", "content": "Here's a comprehensive approach for integrating security testing into CI/CD pipelines:\\n\\n**Pipeline Integration Strategy:**\\n\\n**1. Pre-commit Stage (Developer Workstation)**\\n- **Static analysis**: IDE plugins for real-time vulnerability detection\\n- **Secret scanning**: Git hooks to prevent credential commits\\n- **Dependency checking**: Local SCA tools to catch vulnerable libraries\\n- **Unit security tests**: Security-focused unit tests\\n\\n**2. Commit Stage (CI Pipeline)**\\n- **SAST (Static Application Security Testing)**: SonarQube, Checkmarx, Semgrep\\n- **Secret detection**: TruffleHog, GitLeaks for repository scanning\\n- **License compliance**: FOSSA, WhiteSource for dependency licensing\\n- **Quick wins**: Fast security checks that don't slow down builds\\n\\n**3. Build Stage**\\n- **SCA (Software Composition Analysis)**: Snyk,#!/usr/bin/env python3
"""
Security Training Data Generator
Creates fine-tuning data for security models with optional Azure Storage upload
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import time
import subprocess
import hashlib

# Azure Storage imports (optional)
try:
    from azure.storage.blob import BlobServiceClient
    from azure.identity import DefaultAzureCredential
    AZURE_STORAGE_AVAILABLE = True
except ImportError:
    AZURE_STORAGE_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SecurityTrainingDataGenerator:
    """Generate comprehensive security training data"""
    
    def __init__(self, base_path: str = "./security_data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.training_examples = []
    
    def setup_data_sources(self):
        """Download/clone all security data sources"""
        logger.info("Setting up security data sources...")
        
        sources = {
            'nuclei-templates': 'https://github.com/projectdiscovery/nuclei-templates.git',
            'bugbounty-reports': 'https://github.com/marcotuliocnd/bugbounty-disclosed-reports.git',
            'weird-proxies': 'https://github.com/GrrrDog/weird_proxies.git',
            'bchecks': 'https://github.com/PortSwigger/BChecks.git'
        }
        
        for name, url in sources.items():
            repo_path = self.base_path / name
            if not repo_path.exists():
                logger.info(f"Cloning {name}...")
                try:
                    subprocess.run(['git', 'clone', url, str(repo_path)], check=True)
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to clone {name}: {e}")
            else:
                logger.info(f"Updating {name}...")
                try:
                    subprocess.run(['git', 'pull'], cwd=repo_path, check=True)
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to update {name}: {e}")
        
        # Download RFCs
        self._download_rfcs()
    
    def _download_rfcs(self):
        """Download key security-related RFCs"""
        rfc_dir = self.base_path / "rfcs"
        rfc_dir.mkdir(exist_ok=True)
        
        rfcs = [
            "https://www.ietf.org/rfc/rfc9110.txt",  # HTTP Semantics
            "https://www.ietf.org/rfc/rfc7231.txt",  # HTTP/1.1 Semantics
            "https://www.ietf.org/rfc/rfc7235.txt",  # HTTP/1.1 Authentication
        ]
        
        for rfc_url in rfcs:
            rfc_name = rfc_url.split('/')[-1]
            rfc_path = rfc_dir / rfc_name
            
            if not rfc_path.exists():
                logger.info(f"Downloading {rfc_name}...")
                try:
                    import requests
                    response = requests.get(rfc_url, timeout=30)
                    response.raise_for_status()
                    with open(rfc_path, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    time.sleep(1)  # Be respectful
                except Exception as e:
                    logger.warning(f"Failed to download {rfc_name}: {e}")
    
    def generate_nuclei_examples(self) -> List[Dict]:
        """Generate training examples from Nuclei templates"""
        examples = []
        nuclei_path = self.base_path / "nuclei-templates"
        
        if not nuclei_path.exists():
            logger.warning("Nuclei templates not found. Run --setup first.")
            return examples
        
        logger.info("Processing Nuclei templates...")
        
        # Find YAML files - remove restrictive limits
        yaml_files = list(nuclei_path.rglob("*.yaml")) + list(nuclei_path.rglob("*.yml"))
        yaml_files = [f for f in yaml_files if not any(exclude in str(f).lower() 
                     for exclude in ['.github', 'workflows'])]  # Less restrictive filtering
        
        logger.info(f"Found {len(yaml_files)} Nuclei template files")
        
        # Process more files - increase limit significantly
        for yaml_file in yaml_files[:2000]:  # Increased from 500 to 2000
            try:
                import yaml
                with open(yaml_file, 'r', encoding='utf-8', errors='ignore') as f:
                    template = yaml.safe_load(f)
                
                if not isinstance(template, dict) or 'info' not in template:
                    continue
                
                examples.extend(self._create_nuclei_examples(template, yaml_file))
                
            except Exception as e:
                logger.debug(f"Failed to process {yaml_file}: {e}")
        
        logger.info(f"Generated {len(examples)} Nuclei examples")
        return examples
    
    def _create_nuclei_examples(self, template: Dict, file_path: Path) -> List[Dict]:
        """Create training examples focused on Nuclei template structure and creation"""
        examples = []
        info = template.get('info', {})
        
        name = info.get('name', file_path.stem)
        description = info.get('description', '')
        severity = info.get('severity', 'info')
        tags = info.get('tags', [])
        
        # Get the actual YAML content for template structure learning
        template_yaml = self._get_clean_template_yaml(template)
        
        # Example 1: Template Structure Analysis
        examples.append({
            "messages": [
                {"role": "user", "content": f"Analyze this Nuclei template structure and explain how it works:\\n\\n{template_yaml[:600]}..."},
                {"role": "assistant", "content": f"This Nuclei template follows the standard structure for vulnerability detection:\\n\\n**Template Components:**\\n\\n1. **ID**: `{template.get('id', 'template-id')}` - Unique identifier\\n2. **Info Section**: Contains metadata\\n   - Name: {name}\\n   - Description: {description}\\n   - Severity: {severity}\\n   - Tags: {tags}\\n\\n3. **Detection Logic**: {self._analyze_detection_method(template)}\\n\\n4. **Matchers**: {self._explain_matchers(template)}\\n\\n**How it works:**\\n{self._explain_template_workflow(template)}\\n\\n**Key Learning Points:**\\n- The template uses {self._get_request_method(template)} to interact with targets\\n- Detection relies on {self._get_detection_criteria(template)}\\n- This pattern is effective for {self._get_vulnerability_class(template)} vulnerabilities"}
            ]
        })
        
        # Example 2: Creating Similar Template
        if template_yaml:
            examples.append({
                "messages": [
                    {"role": "user", "content": f"Create a Nuclei template similar to this one but for detecting a different variant of the same vulnerability type:\\n\\n{template_yaml[:400]}..."},
                    {"role": "assistant", "content": f"Based on the structure of the {name} template, here's a variant for detecting similar vulnerabilities:\\n\\n```yaml\\nid: {template.get('id', 'template')}-variant\\n\\ninfo:\\n  name: \\\"{name} - Alternative Detection\\\"\\n  author: \\\"Security Analyst\\\"\\n  severity: {severity}\\n  description: \\\"Variant detection for {description.lower() if description else 'similar vulnerability patterns'}\\\"\\n  tags: {tags}\\n\\n{self._generate_variant_detection_logic(template)}\\n```\\n\\n**Key Changes Made:**\\n{self._explain_variant_changes(template)}\\n\\n**Why This Variant Works:**\\n{self._explain_variant_rationale(template)}\\n\\n**Testing Approach:**\\nValidate this template against known vulnerable instances and ensure it doesn't conflict with the original template's detection scope.\"}
                ]
            })
        
        # Example 3: Template Debugging/Optimization
        examples.append({
            "messages": [
                {"role": "user", "content": f"This Nuclei template is generating false positives. How can I make the detection more precise?\\n\\n{template_yaml[:500]}..."},
                {"role": "assistant", "content": f"To reduce false positives in the {name} template, here are specific optimizations:\\n\\n**Current Detection Issues:**\\n{self._identify_false_positive_sources(template)}\\n\\n**Optimization Strategies:**\\n\\n1. **Refine Matchers:**\\n```yaml\\nmatchers:\\n  - type: dsl\\n    dsl:\\n{self._generate_refined_matchers(template)}\\n```\\n\\n2. **Add Negative Conditions:**\\n```yaml\\nmatchers-condition: and\\nmatchers:\\n  - type: word\\n    words: {self._get_positive_indicators(template)}\\n  - type: word\\n    words: {self._get_negative_indicators(template)}\\n    negative: true\\n```\\n\\n3. **Response Context Analysis:**\\n{self._suggest_context_improvements(template)}\\n\\n**Testing Validation:**\\nTest against both vulnerable and non-vulnerable targets to ensure the refined logic maintains detection capability while reducing false positives.\"}
            ]
        })
        
        return examples
    
    def _get_clean_template_yaml(self, template: Dict) -> str:
        \"\"\"Extract clean YAML representation of template\"\"\"
        try:
            import yaml
            # Create clean version focusing on structure
            clean_template = {
                'id': template.get('id', 'example-template'),
                'info': template.get('info', {}),
            }
            
            # Add the main detection section
            for section in ['http', 'network', 'dns', 'ssl', 'file']:
                if section in template:
                    clean_template[section] = template[section]
                    break
            
            return yaml.dump(clean_template, default_flow_style=False, sort_keys=False)
        except:
            return f\"id: {template.get('id', 'example')}\\ninfo:\\n  name: \\\"{template.get('info', {}).get('name', 'Example')}\\\"\"
    
    def _analyze_detection_method(self, template: Dict) -> str:
        \"\"\"Analyze how the template detects vulnerabilities\"\"\"
        if 'http' in template:
            return \"HTTP requests with response pattern matching\"
        elif 'network' in template:
            return \"Network protocol interaction and response analysis\"
        elif 'dns' in template:
            return \"DNS query and response validation\"
        else:
            return \"Custom detection logic\"
    
    def _explain_matchers(self, template: Dict) -> str:
        \"\"\"Explain the matcher logic\"\"\"
        for section_name in ['http', 'network', 'dns']:
            if section_name in template:
                section = template[section_name]
                if isinstance(section, list) and section:
                    first_item = section[0]
                    if 'matchers' in first_item:
                        matchers = first_item['matchers']
                        if isinstance(matchers, list) and matchers:
                            matcher_types = [m.get('type', 'unknown') for m in matchers if isinstance(m, dict)]
                            return f\"Uses {', '.join(set(matcher_types))} matching\"
        return \"Pattern-based matching logic\"
    
    def _explain_template_workflow(self, template: Dict) -> str:
        \"\"\"Explain how the template executes\"\"\"
        if 'http' in template:
            return \"1. Sends HTTP request(s) to target\\n2. Analyzes response content, headers, and status\\n3. Applies matcher logic to determine vulnerability presence\\n4. Reports findings if conditions are met\"
        elif 'network' in template:
            return \"1. Establishes network connection to target\\n2. Sends protocol-specific probes\\n3. Analyzes network responses\\n4. Matches against vulnerability signatures\"
        else:
            return \"1. Executes detection logic against target\\n2. Collects response data\\n3. Applies matching criteria\\n4. Reports positive matches\"
    
    def _get_request_method(self, template: Dict) -> str:
        \"\"\"Get the request method used\"\"\"
        if 'http' in template:
            return \"HTTP requests\"
        elif 'network' in template:
            return \"network connections\"
        else:
            return \"targeted probes\"
    
    def _get_detection_criteria(self, template: Dict) -> str:
        \"\"\"Get what the template looks for\"\"\"
        # Extract matcher information
        criteria = []
        for section_name in ['http', 'network', 'dns']:
            if section_name in template:
                section = template[section_name]
                if isinstance(section, list):
                    for item in section:
                        if isinstance(item, dict) and 'matchers' in item:
                            for matcher in item['matchers']:
                                if isinstance(matcher, dict):
                                    if 'words' in matcher:
                                        criteria.append(\"specific text patterns\")
                                    if 'regex' in matcher:
                                        criteria.append(\"regular expression patterns\")
                                    if 'status' in matcher:
                                        criteria.append(\"HTTP status codes\")
        
        return ', '.join(set(criteria)) if criteria else \"response pattern analysis\"
    
    def _get_vulnerability_class(self, template: Dict) -> str:
        \"\"\"Determine vulnerability class\"\"\"
        info = template.get('info', {})
        tags = info.get('tags', [])
        name = info.get('name', '').lower()
        
        if any('sqli' in str(tag).lower() for tag in tags) or 'sql' in name:
            return \"SQL injection\"
        elif any('xss' in str(tag).lower() for tag in tags) or 'xss' in name:
            return \"cross-site scripting\"
        elif any('rce' in str(tag).lower() for tag in tags) or 'execution' in name:
            return \"remote code execution\"
        elif any('lfi' in str(tag).lower() for tag in tags) or 'file' in name:
            return \"file inclusion\"
        else:
            return \"general vulnerability\"
    
    def _generate_variant_detection_logic(self, template: Dict) -> str:
        \"\"\"Generate variant detection logic\"\"\"
        if 'http' in template:
            return \"\"\"http:\\n  - method: GET\\n    path:\\n      - \\\"{{BaseURL}}/alternate-endpoint\\\"\\n      - \\\"{{BaseURL}}/variant-path\\\"\\n    \\n    matchers-condition: and\\n    matchers:\\n      - type: status\\n        status:\\n          - 200\\n      - type: word\\n        words:\\n          - \\\"variant-indicator\\\"\\n          - \\\"alternate-pattern\\\"\\n        condition: or\"\"\"
        else:
            return \"\"\"# Add variant-specific detection logic here\\n# Based on the original template structure\"\"\"
    
    def _explain_variant_changes(self, template: Dict) -> str:
        \"\"\"Explain changes made in variant\"\"\"
        return \"\"\"- Modified target paths to check alternative endpoints\\n- Adjusted detection patterns for variant indicators\\n- Updated matcher conditions for broader coverage\\n- Maintained original severity and classification\"\"\"
    
    def _explain_variant_rationale(self, template: Dict) -> str:
        \"\"\"Explain why variant works\"\"\"
        return \"\"\"This variant targets the same vulnerability class but through different attack vectors or endpoints. It complements the original template by covering scenarios the original might miss while maintaining detection accuracy.\"\"\"
    
    def _identify_false_positive_sources(self, template: Dict) -> str:
        \"\"\"Identify potential false positive sources\"\"\"
        return \"\"\"- Generic pattern matching that may trigger on legitimate content\\n- Insufficient response context validation\\n- Missing negative conditions to exclude safe responses\\n- Overly broad matcher conditions\"\"\"
    
    def _generate_refined_matchers(self, template: Dict) -> str:
        \"\"\"Generate more precise matcher logic\"\"\"
        return \"\"\"      - \\\"len(body) > 100 && status_code == 200\\\"\\n      - \\\"contains(tolower(body), 'error') && !contains(tolower(body), 'safe')\\\"\\n      - \\\"response_time < 5000\\\"\"\"\"
    
    def _get_positive_indicators(self, template: Dict) -> List[str]:
        \"\"\"Get positive detection indicators\"\"\"
        return [\"\\\"vulnerability-indicator\\\"\", \"\\\"error-pattern\\\"\"]
    
    def _get_negative_indicators(self, template: Dict) -> List[str]:
        \"\"\"Get negative indicators to exclude\"\"\"
        return [\"\\\"expected-safe-response\\\"\", \"\\\"normal-behavior\\\"\"]
    
    def _suggest_context_improvements(self, template: Dict) -> str:
        \"\"\"Suggest context-based improvements\"\"\"
        return \"\"\"- Check Content-Type header to ensure appropriate response type\\n- Validate response length to avoid tiny/empty responses\\n- Add timing analysis for blind vulnerability detection\\n- Include Server header analysis for technology-specific checks\"\"\""
    
    def _template_to_yaml_string(self, template: Dict) -> str:
        """Convert template dict back to YAML string for examples"""
        try:
            import yaml
            # Create a simplified version for examples
            simplified = {
                'id': template.get('id', 'custom-template'),
                'info': template.get('info', {}),
            }
            
            # Add the main detection section
            for section in ['http', 'network', 'dns', 'ssl']:
                if section in template:
                    simplified[section] = template[section]
                    break
            
            return yaml.dump(simplified, default_flow_style=False, sort_keys=False)
        except:
            return "# Template structure would go here"
    
    def _get_template_scope(self, template: Dict) -> str:
        """Determine what the template tests"""
        if 'http' in template:
            return "web applications and HTTP services"
        elif 'network' in template:
            return "network services and protocols"
        elif 'dns' in template:
            return "DNS services and configurations"
        elif 'ssl' in template:
            return "SSL/TLS configurations"
        else:
            return "general security configurations"
    
    def _get_detection_method(self, template: Dict) -> str:
        """Describe how the template detects vulnerabilities"""
        if 'http' in template:
            http_section = template['http']
            if isinstance(http_section, list) and http_section:
                first_request = http_section[0]
                if 'matchers' in first_request:
                    return "HTTP request/response pattern matching"
                else:
                    return "HTTP response analysis"
        elif 'network' in template:
            return "Network protocol interaction and response analysis"
        else:
            return "Pattern-based detection and response analysis"
    
    def _assess_false_positive_risk(self, template: Dict) -> str:
        """Assess false positive risk"""
        # This is a simplified assessment
        severity = template.get('info', {}).get('severity', 'info')
        
        if severity in ['critical', 'high']:
            return "Low (high-confidence detection patterns)"
        elif severity == 'medium':
            return "Medium (requires manual verification)"
        else:
            return "Medium-High (may require additional validation)"
    
    def _get_usage_timing(self, tags: List, severity: str) -> str:
        """Suggest when to use this template"""
        if severity in ['critical', 'high']:
            return "Early in testing - high priority for immediate validation"
        elif 'discovery' in str(tags).lower():
            return "Initial reconnaissance phase"
        elif any(word in str(tags).lower() for word in ['exploit', 'rce']):
            return "After initial discovery - when target validation is needed"
        else:
            return "During comprehensive scanning phase"
    
    def _suggest_complementary_tools(self, tags: List) -> str:
        """Suggest complementary tools"""
        tag_str = str(tags).lower()
        
        tools = []
        if 'sqli' in tag_str:
            tools.append("SQLMap for exploitation validation")
        if 'xss' in tag_str:
            tools.append("Manual browser testing")
        if 'rce' in tag_str:
            tools.append("Manual payload testing")
        if 'discovery' in tag_str:
            tools.append("Nmap for service enumeration")
        
        tools.append("Burp Suite for manual validation")
        
        return ", ".join(tools)
    
    def _suggest_validation_steps(self, template: Dict) -> str:
        """Suggest validation steps"""
        steps = [
            "Manual verification of findings",
            "Cross-check with other scanning tools",
        ]
        
        if 'http' in template:
            steps.append("Test with different HTTP methods and parameters")
        
        steps.append("Document findings with clear evidence")
        
        return "; ".join(steps)
    
    def _get_template_context(self, tags: List, name: str) -> str:
        """Get context for when to use a template"""
        if any('cve' in str(tag).lower() for tag in tags):
            return "known CVE vulnerabilities"
        elif any('sqli' in str(tag).lower() for tag in tags):
            return "SQL injection vulnerabilities"
        elif any('xss' in str(tag).lower() for tag in tags):
            return "cross-site scripting vulnerabilities"
        elif any('rce' in str(tag).lower() for tag in tags):
            return "remote code execution vulnerabilities"
        elif 'config' in name.lower():
            return "configuration issues and misconfigurations"
        else:
            return "security assessments and vulnerability discovery"
    
    def generate_bugbounty_examples(self) -> List[Dict]:
        """Generate training examples from bug bounty reports"""
        examples = []
        bugbounty_path = self.base_path / "bugbounty-reports"
        
        if not bugbounty_path.exists():
            logger.warning("Bug bounty reports not found. Run --setup first.")
            return examples
        
        logger.info("Processing bug bounty reports...")
        
        md_files = list(bugbounty_path.rglob("*.md"))
        logger.info(f"Found {len(md_files)} bug bounty report files")
        
        # Increase limit significantly
        for md_file in md_files[:500]:  # Increased from 100 to 500
            try:
                with open(md_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                examples.extend(self._create_bugbounty_examples(content, md_file))
                
            except Exception as e:
                logger.debug(f"Failed to process {md_file}: {e}")
        
        logger.info(f"Generated {len(examples)} bug bounty examples")
        return examples
    
    def _create_bugbounty_examples(self, content: str, file_path: Path) -> List[Dict]:
        """Create training examples from bug bounty report content"""
        examples = []
        
        # Extract vulnerability type from content
        vuln_type = self._identify_vulnerability_type(content)
        
        if vuln_type:
            # Methodology example
            examples.append({
                "messages": [
                    {"role": "user", "content": f"What's the best methodology for finding {vuln_type} vulnerabilities based on disclosed bug bounty reports?"},
                    {"role": "assistant", "content": f"Based on successful bug bounty disclosures for {vuln_type}, here's an effective approach:\\n\\n**1. Reconnaissance Phase:**\\n- Map out the application structure and identify potential entry points\\n- Look for parameters that accept user input\\n\\n**2. Testing Strategy:**\\n- Start with automated scanning to identify obvious cases\\n- Follow up with manual testing for edge cases\\n- Focus on areas where {vuln_type} commonly occurs\\n\\n**3. Validation:**\\n- Develop clear proof-of-concept demonstrations\\n- Document the impact and potential business consequences\\n- Test in a controlled manner to avoid causing damage\\n\\nReal-world bug bounty hunters often succeed by being persistent and thinking creatively about how vulnerabilities might manifest in different contexts."}
                ]
            })
            
            # Tool integration example
            examples.append({
                "messages": [
                    {"role": "user", "content": f"How can I use Burp Suite to efficiently test for {vuln_type}?"},
                    {"role": "assistant", "content": f"Here's how to leverage Burp Suite for {vuln_type} testing:\\n\\n**Setup:**\\n- Configure browser proxy through Burp\\n- Enable passive and active scanning\\n\\n**Manual Testing:**\\n- Use Repeater to modify requests and test different payloads\\n- Leverage Intruder for systematic parameter testing\\n- Monitor responses for indicators of {vuln_type}\\n\\n**Automation:**\\n- Configure custom scan profiles focused on {vuln_type}\\n- Use extensions that specialize in this vulnerability class\\n- Set up active scan rules for comprehensive coverage\\n\\nThe key is combining Burp's automation with manual creativity - many bug bounty findings come from manual testing that automated tools miss."}
                ]
            })
        
        return examples
    
    def generate_bchecks_examples(self) -> List[Dict]:
        """Generate training examples from Burp BChecks"""
        examples = []
        bchecks_path = self.base_path / "bchecks"
        
        if not bchecks_path.exists():
            logger.warning("BChecks not found. Run --setup first.")
            return examples
        
        logger.info("Processing Burp BChecks...")
        
        # Find all BCheck files
        bcheck_files = list(bchecks_path.rglob("*.bcheck"))
        logger.info(f"Found {len(bcheck_files)} BCheck files")
        
        # Increase limit significantly  
        for bcheck_file in bcheck_files[:800]:  # Increased from 200 to 800
            try:
                with open(bcheck_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                examples.extend(self._create_bcheck_examples(content, bcheck_file))
                
            except Exception as e:
                logger.debug(f"Failed to process {bcheck_file}: {e}")
        
        logger.info(f"Generated {len(examples)} BChecks examples")
        return examples
    
    def _create_bcheck_examples(self, content: str, file_path: Path) -> List[Dict]:
        """Create training examples focused on BCheck structure and creation"""
        examples = []
        
        # Extract BCheck metadata and structure
        metadata = self._parse_bcheck_metadata(content)
        bcheck_structure = self._extract_bcheck_structure(content)
        
        if not metadata:
            return examples
        
        # Example 1: BCheck Structure Analysis
        examples.append({
            "messages": [
                {"role": "user", "content": f"Analyze this BCheck structure and explain how it works:\\n\\n{content[:800]}..."},
                {"role": "assistant", "content": f"This BCheck follows the standard Burp Suite BCheck format:\\n\\n**Metadata Section:**\\n```\\nmetadata:\\n    language: {metadata.get('language', 'v1-beta')}\\n    name: \\\"{metadata.get('name', 'Unknown')}\\\"\\n    description: \\\"{metadata.get('description', 'No description')}\\\"\\n    author: \\\"{metadata.get('author', 'Unknown')}\\\"\\n    tags: {metadata.get('tags', [])}\\n```\\n\\n**Detection Logic:**\\n{self._explain_bcheck_logic(bcheck_structure)}\\n\\n**How it works:**\\n1. **Trigger condition**: {bcheck_structure.get('trigger', 'Request-based trigger')}\\n2. **Request manipulation**: {bcheck_structure.get('request_logic', 'Modifies request parameters')}\\n3. **Response analysis**: {bcheck_structure.get('response_analysis', 'Analyzes response patterns')}\\n4. **Issue reporting**: {bcheck_structure.get('reporting', 'Reports vulnerability if conditions met')}\\n\\n**Key Components:**\\n- **Given/Then logic**: Conditional execution based on request/response\\n- **Matchers**: Pattern matching for vulnerability detection\\n- **Variables**: Dynamic content extraction and reuse"}
            ]
        })
        
        # Example 2: Creating Custom BCheck
        examples.append({
            "messages": [
                {"role": "user", "content": f"Create a custom BCheck similar to this one but for a different vulnerability type:\\n\\n{content[:600]}..."},
                {"role": "assistant", "content": f"Based on the structure of the {metadata.get('name', 'original')} BCheck, here's a custom variant:\\n\\n```\\nmetadata:\\n    language: v1-beta\\n    name: \\\"Custom {metadata.get('name', 'Security Check')} Variant\\\"\\n    description: \\\"Enhanced detection for {self._get_variant_description(metadata)}\\\"\\n    author: \\\"Security Analyst\\\"\\n    tags: {self._get_variant_tags(metadata)}\\n\\n{self._generate_custom_bcheck_logic(bcheck_structure, metadata)}\\n```\\n\\n**Key Modifications:**\\n{self._explain_bcheck_modifications(metadata)}\\n\\n**Why This Variant Works:**\\n{self._explain_bcheck_variant_rationale(metadata)}\\n\\n**Testing Strategy:**\\nTest this BCheck against both vulnerable and non-vulnerable endpoints to ensure accuracy and minimal false positives."}
            ]
        })
        
        # Example 3: BCheck Optimization
        examples.append({
            "messages": [
                {"role": "user", "content": f"This BCheck is generating false positives. How can I optimize the detection logic?\\n\\n{content[:500]}..."},
                {"role": "assistant", "content": f"To optimize the {metadata.get('name', 'BCheck')} and reduce false positives:\\n\\n**Current Issues Analysis:**\\n{self._analyze_bcheck_false_positives(bcheck_structure)}\\n\\n**Optimization Strategies:**\\n\\n1. **Refine Response Matching:**\\n```\\nif {{latest.response.status_code}} is \\\"200\\\" and\\n   {{latest.response.body}} matches \\\"specific-error-pattern\\\" and\\n   {{latest.response.body}} does not match \\\"safe-indicator\\\" and\\n   {{latest.response.headers}} contains \\\"vulnerable-header\\\" then\\n```\\n\\n2. **Add Context Validation:**\\n```\\nif {{base.request.url}} matches \\\".*/admin/.*\\\" and\\n   {{latest.response.body}} length > 100 then\\n```\\n\\n3. **Implement Negative Conditions:**\\n{self._suggest_negative_conditions(metadata)}\\n\\n**Enhanced Detection Logic:**\\n{self._generate_optimized_bcheck_logic(bcheck_structure)}\\n\\n**Validation Approach:**\\nTest against known false positive scenarios and ensure the refined logic maintains detection capability while improving precision."}
            ]
        })
        
        # Example 4: BCheck Integration Workflow
        examples.append({
            "messages": [
                {"role": "user", "content": f"How do I integrate the {metadata.get('name', 'BCheck')} into my Burp Suite testing workflow effectively?"},
                {"role": "assistant", "content": f"Here's how to effectively integrate the {metadata.get('name', 'BCheck')} into your testing workflow:\\n\\n**Pre-Testing Setup:**\\n1. **Import BCheck**: Extensions → BChecks → Load from file\\n2. **Configure scope**: Ensure target scope includes relevant endpoints\\n3. **Adjust settings**: Configure scan speed and thoroughness\\n\\n**Integration Strategy:**\\n\\n**During Active Scanning:**\\n- Enable BChecks in scan configuration\\n- Focus on endpoints where {self._get_bcheck_target_context(metadata)}\\n- Monitor for findings in real-time\\n\\n**Manual Testing Integration:**\\n- Use alongside manual parameter testing\\n- Validate BCheck findings manually in Repeater\\n- Combine with Intruder for systematic testing\\n\\n**Workflow Optimization:**\\n{self._suggest_workflow_optimization(metadata)}\\n\\n**Results Analysis:**\\n- Review findings in Issues tab\\n- Cross-reference with other tool results\\n- Validate exploitability manually\\n- Document findings with clear evidence\\n\\n**Continuous Improvement:**\\n- Track false positive patterns\\n- Refine BCheck logic based on results\\n- Share findings with team for validation\\n\\nThis BCheck is most effective when combined with {self._suggest_complementary_testing_approach(metadata)}."}
            ]
        })
        
        return examples
    
    def _extract_bcheck_structure(self, content: str) -> Dict:
        \"\"\"Extract the structural elements of a BCheck\"\"\"
        structure = {}
        
        lines = content.split('\\n')
        current_section = None
        
        for line in lines:
            line_stripped = line.strip()
            
            if line_stripped.startswith('given'):
                structure['trigger'] = 'Request-based conditional trigger'
                current_section = 'given'
            elif line_stripped.startswith('if '):
                if 'conditions' not in structure:
                    structure['conditions'] = []
                structure['conditions'].append(line_stripped)
            elif line_stripped.startswith('send request'):
                structure['request_logic'] = 'Custom request generation and manipulation'
            elif line_stripped.startswith('report issue'):
                structure['reporting'] = 'Structured vulnerability reporting'
            elif 'response' in line_stripped and 'matches' in line_stripped:
                structure['response_analysis'] = 'Pattern-based response analysis'
        
        return structure
    
    def _explain_bcheck_logic(self, structure: Dict) -> str:
        \"\"\"Explain the BCheck logic flow\"\"\"
        explanation = \"The BCheck uses conditional logic:\\n\"
        
        if structure.get('trigger'):
            explanation += f\"- **Trigger**: {structure['trigger']}\\n\"
        if structure.get('conditions'):
            explanation += f\"- **Conditions**: {len(structure['conditions'])} conditional checks\\n\"
        if structure.get('request_logic'):
            explanation += f\"- **Request handling**: {structure['request_logic']}\\n\"
        if structure.get('response_analysis'):
            explanation += f\"- **Response analysis**: {structure['response_analysis']}\\n\"
        
        return explanation
    
    def _get_variant_description(self, metadata: Dict) -> str:
        \"\"\"Generate description for variant BCheck\"\"\"
        original_desc = metadata.get('description', 'security testing')
        return f\"variant patterns related to {original_desc}\"
    
    def _get_variant_tags(self, metadata: Dict) -> List[str]:
        \"\"\"Generate tags for variant BCheck\"\"\"
        original_tags = metadata.get('tags', [])
        variant_tags = original_tags.copy() if original_tags else []
        variant_tags.append('custom')
        return variant_tags
    
    def _generate_custom_bcheck_logic(self, structure: Dict, metadata: Dict) -> str:
        \"\"\"Generate custom BCheck logic\"\"\"
        return \"\"\"given request then\\n    if {{base.request.url}} matches \\\".*\\/custom-endpoint\\/.*\\\" then\\n        send request called check:\\n            replacing path with \\\"/custom-test-path\\\"\\n            \\n        if {{check.response.status_code}} is \\\"200\\\" and\\n           {{check.response.body}} matches \\\"custom-vulnerability-indicator\\\" then\\n            report issue:\\n                severity: medium\\n                confidence: tentative\\n                detail: \\\"Custom vulnerability detected in endpoint\\\"\\n                remediation: \\\"Implement proper input validation and output encoding\\\"\"\"\"
    
    def _explain_bcheck_modifications(self, metadata: Dict) -> str:
        \"\"\"Explain modifications made to BCheck\"\"\"
        return \"\"\"- Modified target URL pattern for custom endpoints\\n- Adjusted vulnerability indicators for specific application\\n- Updated severity and confidence levels\\n- Added custom remediation guidance\"\"\"
    
    def _explain_bcheck_variant_rationale(self, metadata: Dict) -> str:
        \"\"\"Explain rationale for BCheck variant\"\"\"
        return \"\"\"This variant targets similar vulnerability patterns but in different application contexts. It maintains the core detection logic while adapting to specific environmental requirements.\"\"\"
    
    def _analyze_bcheck_false_positives(self, structure: Dict) -> str:
        \"\"\"Analyze potential false positive sources\"\"\"
        return \"\"\"- Overly broad URL matching patterns\\n- Generic response content matching\\n- Insufficient negative condition checks\\n- Missing context validation for legitimate responses\"\"\"
    
    def _suggest_negative_conditions(self, metadata: Dict) -> str:
        \"\"\"Suggest negative conditions to reduce false positives\"\"\"
        return \"\"\"```\\nand {{latest.response.body}} does not match \\\"expected-safe-content\\\"\\nand {{latest.response.headers}} does not contain \\\"safe-indicator-header\\\"\\nand {{latest.response.body}} length > 50\\n```\"\"\"
    
    def _generate_optimized_bcheck_logic(self, structure: Dict) -> str:
        \"\"\"Generate optimized BCheck logic\"\"\"
        return \"\"\"Enhanced logic with multiple validation layers:\\n- URL pattern specificity\\n- Response content validation\\n- Header analysis\\n- Content length checks\\n- Timing validation\"\"\"
    
    def _get_bcheck_target_context(self, metadata: Dict) -> str:
        \"\"\"Get context for where BCheck should be used\"\"\"
        tags = metadata.get('tags', [])
        if any('admin' in str(tag).lower() for tag in tags):
            return \"administrative interfaces and privileged functionality\"
        elif any('api' in str(tag).lower() for tag in tags):
            return \"API endpoints and REST services\"
        else:
            return \"dynamic web application endpoints\"
    
    def _suggest_workflow_optimization(self, metadata: Dict) -> str:
        \"\"\"Suggest workflow optimization for BCheck\"\"\"
        return \"\"\"- **Scope targeting**: Focus on relevant application areas\\n- **Timing coordination**: Run during comprehensive scan phases\\n- **Result correlation**: Cross-check with other security tools\\n- **Manual validation**: Always verify findings manually\"\"\"
    
    def _suggest_complementary_testing_approach(self, metadata: Dict) -> str:
        \"\"\"Suggest complementary testing approaches\"\"\"
        return \"\"\"manual parameter testing, Nuclei template scanning, and traditional Burp Suite active scanning\"\"\""
    
    def _parse_bcheck_metadata(self, content: str) -> Dict:
        """Parse BCheck metadata from content"""
        metadata = {}
        
        lines = content.split('\\n')
        in_metadata = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('metadata:'):
                in_metadata = True
                continue
            elif in_metadata and line and not line.startswith(' ') and ':' not in line:
                break
            elif in_metadata and ':' in line:
                try:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip().strip('\"\'')
                    
                    if key == 'tags':
                        # Handle tags as list
                        if value.startswith('[') and value.endswith(']'):
                            # Parse list format: [tag1, tag2]
                            tags_str = value.strip('[]')
                            metadata['tags'] = [tag.strip().strip('\"\'') for tag in tags_str.split(',') if tag.strip()]
                        else:
                            # Single tag
                            metadata['tags'] = [value]
                    else:
                        metadata[key] = value
                except ValueError:
                    continue
        
        return metadata
    
    def _get_bcheck_detection_focus(self, metadata: Dict) -> str:
        """Get what the BCheck focuses on detecting"""
        tags = metadata.get('tags', [])
        name = metadata.get('name', '').lower()
        
        if any('sqli' in str(tag).lower() for tag in tags) or 'sql' in name:
            return 'SQL injection vulnerabilities'
        elif any('xss' in str(tag).lower() for tag in tags) or 'xss' in name:
            return 'cross-site scripting vulnerabilities'
        elif any('ssrf' in str(tag).lower() for tag in tags) or 'ssrf' in name:
            return 'server-side request forgery vulnerabilities'
        elif any('rce' in str(tag).lower() for tag in tags) or 'execution' in name:
            return 'remote code execution vulnerabilities'
        elif any('auth' in str(tag).lower() for tag in tags) or 'auth' in name:
            return 'authentication and authorization issues'
        elif any('info' in str(tag).lower() for tag in tags) or 'disclosure' in name:
            return 'information disclosure vulnerabilities'
        else:
            return 'security vulnerabilities and misconfigurations'
    
    def _get_bcheck_use_case(self, metadata: Dict) -> str:
        """Get the primary use case for this BCheck"""
        tags = metadata.get('tags', [])
        
        if any('api' in str(tag).lower() for tag in tags):
            return 'API endpoints and REST services'
        elif any('web' in str(tag).lower() for tag in tags):
            return 'web applications and dynamic content'
        elif any('admin' in str(tag).lower() for tag in tags):
            return 'administrative interfaces and privileged areas'
        elif any('file' in str(tag).lower() for tag in tags):
            return 'file upload and management functionality'
        else:
            return 'general web application security testing'
    
    def _suggest_complementary_testing(self, metadata: Dict) -> str:
        """Suggest complementary testing approaches"""
        tags = metadata.get('tags', [])
        
        if any('sqli' in str(tag).lower() for tag in tags):
            return 'manual SQL injection testing and SQLMap validation'
        elif any('xss' in str(tag).lower() for tag in tags):
            return 'manual payload testing and DOM analysis'
        elif any('auth' in str(tag).lower() for tag in tags):
            return 'privilege escalation testing and session analysis'
        else:
            return 'manual verification and additional automated scanning'
        """Identify vulnerability type from bug bounty report content"""
        content_lower = content.lower()
        
        vuln_patterns = {
            'SQL injection': ['sql', 'sqli', 'injection', 'database'],
            'Cross-site scripting': ['xss', 'cross-site', 'script'],
            'Remote code execution': ['rce', 'remote code', 'command injection'],
            'Server-side request forgery': ['ssrf', 'server-side request'],
            'Insecure direct object reference': ['idor', 'insecure direct'],
            'Cross-site request forgery': ['csrf', 'cross-site request'],
            'Local file inclusion': ['lfi', 'local file', 'path traversal'],
            'Authentication bypass': ['bypass', 'authentication', 'privilege escalation']
        }
        
        for vuln_type, keywords in vuln_patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                return vuln_type
        
        return None
    
            {
                "messages": [
                    {"role": "user", "content": "What's the best approach for integrating security testing into a CI/CD pipeline?"},
                    {"role": "assistant", "content": "Here's a comprehensive approach for integrating security testing into CI/CD pipelines:\\n\\n**Pipeline Integration Strategy:**\\n\\n**1. Pre-commit Stage (Developer Workstation)**\\n- **Static analysis**: IDE plugins for real-time vulnerability detection\\n- **Secret scanning**: Git hooks to prevent credential commits\\n- **Dependency checking**: Local SCA tools to catch vulnerable libraries\\n- **Unit security tests**: Security-focused unit tests\\n\\n**2. Commit Stage (CI Pipeline)**\\n- **SAST (Static Application Security Testing)**: SonarQube, Checkmarx, Semgrep\\n- **Secret detection**: TruffleHog, GitLeaks for repository scanning\\n- **License compliance**: FOSSA, WhiteSource for dependency licensing\\n- **Quick wins**: Fast security checks that don't slow down builds\\n\\n**3. Build Stage**\\n- **SCA (Software Composition Analysis)**: Snyk, OWASP Dependency-Check\\n- **Container scanning**: Trivy, Clair for Docker image vulnerabilities\\n- **Infrastructure as Code**: Checkov, Terrascan for Terraform/CloudFormation\\n- **Binary analysis**: Quick malware/tampering checks\\n\\n**4. Test Stage (Dynamic Testing)**\\n- **DAST (Dynamic Application Security Testing)**: OWASP ZAP, Burp Suite Enterprise\\n- **API testing**: Automated API security testing with Nuclei templates\\n- **Authentication testing**: Automated login and session management tests\\n- **Performance security**: Load testing with security scenarios\\n\\n**5. Staging/Pre-production**\\n- **Comprehensive DAST**: Full application security scanning\\n- **Penetration testing**: Automated pen-testing tools\\n- **Configuration assessment**: CIS benchmarks, security baselines\\n- **Compliance checking**: SOC2, PCI DSS automated compliance tests\\n\\n**6. Production (Continuous Monitoring)**\\n- **Runtime security**: RASP (Runtime Application Self-Protection)\\n- **Behavioral analysis**: Anomaly detection and threat monitoring\\n- **Vulnerability management**: Continuous scanning and patch management\\n- **Security metrics**: KPIs and dashboard monitoring\\n\\n**Implementation Best Practices:**\\n\\n**Pipeline Design:**\\n- **Fail fast**: Critical security issues should stop the pipeline\\n- **Graduated gates**: Different severity thresholds for different stages\\n- **Parallel execution**: Run security tests in parallel to minimize delay\\n- **Caching**: Cache scan results for unchanged components\\n\\n**Tool Configuration:**\\n- **Baseline establishment**: Define acceptable risk levels\\n- **False positive management**: Tune tools to reduce noise\\n- **Policy as code**: Version control security policies\\n- **Custom rules**: Create organization-specific security rules\\n\\n**Integration Examples:**\\n\\n**GitLab CI/CD:**\\n```yaml\\nstages:\\n  - security-sast\\n  - build\\n  - security-dast\\n  - deploy\\n\\nsast:\\n  stage: security-sast\\n  script:\\n    - semgrep --config=auto --json --output=sast-report.json .\\n  artifacts:\\n    reports:\\n      sast: sast-report.json\\n\\ncontainer_scan:\\n  stage: build\\n  script:\\n    - trivy image --format json --output container-report.json $CI_REGISTRY_IMAGE\\n  artifacts:\\n    reports:\\n      container_scanning: container-report.json\\n\\ndast:\\n  stage: security-dast\\n  script:\\n    - zap-baseline.py -t $APPLICATION_URL -J dast-report.json\\n  artifacts:\\n    reports:\\n      dast: dast-report.json\\n```\\n\\n**GitHub Actions:**\\n```yaml\\nname: Security Pipeline\\non: [push, pull_request]\\n\\njobs:\\n  security-scan:\\n    runs-on: ubuntu-latest\\n    steps:\\n      - uses: actions/checkout@v2\\n      \\n      - name: Run Semgrep\\n        uses: returntocorp/semgrep-action@v1\\n        with:\\n          config: auto\\n          \\n      - name: Run Snyk\\n        uses: snyk/actions/node@master\\n        env:\\n          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}\\n          \\n      - name: Container Security Scan\\n        run: |\\n          docker build -t myapp .\\n          trivy image myapp\\n```\\n\\n**Metrics and KPIs:**\\n- **Mean Time to Detection (MTTD)**: How quickly vulnerabilities are found\\n- **Mean Time to Resolution (MTTR)**: How quickly vulnerabilities are fixed\\n- **Security test coverage**: Percentage of code/components tested\\n- **False positive rate**: Accuracy of security tools\\n- **Compliance score**: Adherence to security policies\\n\\n**Cultural Integration:**\\n- **Security champions**: Train developers in security practices\\n- **Shift-left mentality**: Make security everyone's responsibility\\n- **Feedback loops**: Provide actionable security feedback to developers\\n- **Training**: Regular security training and awareness programs\\n\\n**Tool Recommendations by Stage:**\\n\\n**SAST Tools:**\\n- **Open source**: Semgrep, Bandit, ESLint security plugins\\n- **Commercial**: Checkmarx, Veracode, Fortify\\n\\n**DAST Tools:**\\n- **Open source**: OWASP ZAP, Nuclei\\n- **Commercial**: Burp Suite Enterprise, Rapid7 AppSpider\\n\\n**Container Security:**\\n- **Open source**: Trivy, Clair\\n- **Commercial**: Twistlock, Aqua Security\\n\\n**Dependency Scanning:**\\n- **Open source**: OWASP Dependency-Check, Retire.js\\n- **Commercial**: Snyk, WhiteSource\\n\\nThis approach ensures security is embedded throughout the development lifecycle while maintaining development velocity and providing actionable feedback to teams."}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "How do I set up a security testing lab for practicing web application penetration testing?"},
                    {"role": "assistant", "content": "Here's a comprehensive guide for setting up a security testing lab:\\n\\n**Lab Infrastructure Options:**\\n\\n**Option 1: Local Virtual Machine Setup**\\n- **Hypervisor**: VMware Workstation, VirtualBox, or Parallels\\n- **Advantages**: Complete control, offline testing, no internet dependency\\n- **Requirements**: 16GB+ RAM, 500GB+ storage, modern CPU with virtualization\\n\\n**Option 2: Cloud-Based Lab**\\n- **Platforms**: AWS, Azure, GCP with isolated VPCs\\n- **Advantages**: Scalable, accessible anywhere, easy to reset\\n- **Considerations**: Cost management, proper network isolation\\n\\n**Option 3: Container-Based Lab**\\n- **Technology**: Docker, Docker Compose, Kubernetes\\n- **Advantages**: Lightweight, version controlled, easy deployment\\n- **Tools**: docker-compose files for multi-service applications\\n\\n**Vulnerable Applications for Practice:**\\n\\n**Web Applications:**\\n- **DVWA (Damn Vulnerable Web Application)**: Classic beginner-friendly app\\n- **bWAPP**: Comprehensive vulnerability coverage\\n- **WebGoat**: OWASP's interactive teaching platform\\n- **Juice Shop**: Modern single-page application with OWASP Top 10\\n- **Mutillidae**: Deliberately vulnerable PHP application\\n- **VulnHub VMs**: Various downloadable vulnerable systems\\n\\n**Network/Infrastructure:**\\n- **Metasploitable 2/3**: Vulnerable Linux distributions\\n- **VulnNet series**: CTF-style vulnerable networks\\n- **HackTheBox retired machines**: Professional-grade challenges\\n- **TryHackMe rooms**: Guided learning scenarios\\n\\n**API Testing:**\\n- **vAPI**: Vulnerable API application\\n- **Pixi**: Vulnerable GraphQL API\\n- **crAPI**: Completely Ridiculous API for testing\\n\\n**Container Security:**\\n- **Damn Vulnerable Container**: Docker security testing\\n- **Kubernetes Goat**: K8s security scenarios\\n\\n**Essential Security Tools Setup:**\\n\\n**Operating System:**\\n- **Kali Linux**: Pre-configured with most security tools\\n- **Parrot Security OS**: Alternative penetration testing distribution\\n- **Ubuntu/Debian**: Custom setup with manual tool installation\\n\\n**Core Tools Installation:**\\n```bash\\n# Web Application Testing\\nsudo apt update && sudo apt install -y \\\\\\n    burpsuite \\\\\\n    zaproxy \\\\\\n    sqlmap \\\\\\n    nikto \\\\\\n    dirb \\\\\\n    gobuster \\\\\\n    wfuzz\\n\\n# Network Testing\\nsudo apt install -y \\\\\\n    nmap \\\\\\n    masscan \\\\\\n    metasploit-framework \\\\\\n    john \\\\\\n    hashcat \\\\\\n    hydra\\n\\n# Modern Tools\\npip3 install nuclei-templates\\ngo install -v github.com/projectdiscovery/nuclei/v2/cmd/nuclei@latest\\ngo install github.com/ffuf/ffuf@latest\\n```\\n\\n**Lab Network Architecture:**\\n\\n**Isolated Network Design:**\\n```\\nAttacker Machine (Kali)  →  DMZ Network  →  Vulnerable Applications\\n     |                           |              |\\n   Tools                   Monitoring      Target Apps\\n   Burp Suite             Traffic Analysis  DVWA\\n   Nuclei                 Wireshark        Juice Shop\\n   Nmap                   Logs             Custom Apps\\n```\\n\\n**Docker Compose Lab Example:**\\n```yaml\\nversion: '3.8'\\nservices:\\n  dvwa:\\n    image: vulnerables/web-dvwa\\n    ports:\\n      - \\\"80:80\\\"\\n    environment:\\n      - MYSQL_ROOT_PASSWORD=password\\n      \\n  juiceshop:\\n    image: bkimminich/juice-shop\\n    ports:\\n      - \\\"3000:3000\\\"\\n      \\n  webgoat:\\n    image: webgoat/goatandwolf\\n    ports:\\n      - \\\"8080:8080\\\"\\n      - \\\"9090:9090\\\"\\n      \\n  mutillidae:\\n    image: citizenstig/nowasp\\n    ports:\\n      - \\\"8081:80\\\"\\n\\n  kali:\\n    image: kalilinux/kali-rolling\\n    volumes:\\n      - ./tools:/tools\\n    tty: true\\n    stdin_open: true\\n```\\n\\n**Lab Management and Workflow:**\\n\\n**Snapshot Management:**\\n- **Clean baselines**: Create snapshots of fresh vulnerable applications\\n- **Progress saves**: Save states at different testing phases\\n- **Reset capability**: Quick restoration for repeated practice\\n\\n**Documentation Setup:**\\n- **Note-taking tools**: Obsidian, Notion, or simple markdown\\n- **Screenshot tools**: Flameshot, Greenshot for evidence capture\\n- **Report templates**: Professional penetration testing report formats\\n\\n**Practice Scenarios:**\\n\\n**Beginner Level:**\\n1. **Information gathering**: Subdomain enumeration, technology identification\\n2. **Basic vulnerabilities**: SQL injection, XSS, directory traversal\\n3. **Authentication bypass**: Weak passwords, session management\\n4. **File upload vulnerabilities**: Unrestricted file upload testing\\n\\n**Intermediate Level:**\\n1. **Advanced injection**: Blind SQL injection, NoSQL injection\\n2. **Privilege escalation**: Horizontal and vertical access control\\n3. **Business logic flaws**: Application-specific vulnerabilities\\n4. **API security**: REST API testing, GraphQL vulnerabilities\\n\\n**Advanced Level:**\\n1. **Exploit chaining**: Combining multiple vulnerabilities\\n2. **Binary exploitation**: Buffer overflows, format string bugs\\n3. **Advanced persistent threats**: Multi-stage attack simulation\\n4. **Zero-day research**: Finding unknown vulnerabilities\\n\\n**Learning Path and Certifications:**\\n\\n**Structured Learning:**\\n- **OWASP Testing Guide**: Follow systematic testing methodology\\n- **PortSwigger Academy**: Free comprehensive web security training\\n- **PentesterLab**: Hands-on vulnerable applications with walkthroughs\\n- **HackTheBox Academy**: Structured cybersecurity learning paths\\n\\n**Certification Preparation:**\\n- **OSCP**: Offensive Security Certified Professional\\n- **CEH**: Certified Ethical Hacker\\n- **CISSP**: Information systems security certification\\n- **GWEB**: GIAC Web Application Penetration Tester\\n\\n**Lab Maintenance:**\\n\\n**Regular Updates:**\\n- **Tool updates**: Keep security tools current with latest signatures\\n- **Vulnerable apps**: Update to newer versions for fresh challenges\\n- **Documentation**: Maintain testing methodologies and findings\\n\\n**Security Considerations:**\\n- **Network isolation**: Ensure lab traffic doesn't reach production\\n- **Legal compliance**: Only test on systems you own or have permission\\n- **Data protection**: No real sensitive data in lab environments\\n\\n**Advanced Lab Features:**\\n\\n**Monitoring and Analysis:**\\n- **Traffic capture**: Wireshark for network analysis\\n- **Log analysis**: ELK stack for security event correlation\\n- **Threat simulation**: Caldera or Atomic Red Team for attack simulation\\n\\n**Automation Integration:**\\n- **CI/CD testing**: Integrate automated security testing pipelines\\n- **Custom tools**: Develop custom exploits and testing tools\\n- **Reporting automation**: Automated vulnerability report generation\\n\\nThis lab setup provides a comprehensive environment for developing and practicing web application penetration testing skills in a safe, controlled environment."}
                ]
            }
        ]
        
        return examples
    
    def generate_complete_dataset(self) -> List[Dict]:
        """Generate the complete training dataset"""
        logger.info("Generating comprehensive security training dataset...")
        
        all_examples = []
        
        # Generate examples from different sources
        all_examples.extend(self.generate_nuclei_examples())
        all_examples.extend(self.generate_bugbounty_examples())
        all_examples.extend(self.generate_bchecks_examples())
        all_examples.extend(self.generate_general_security_examples())
        
        logger.info(f"Generated {len(all_examples)} total training examples")
        return all_examples
    
    def save_dataset(self, examples: List[Dict], output_path: str) -> str:
        """Save dataset in Azure OpenAI fine-tuning format"""
        output_file = Path(output_path)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\\n')
        
        logger.info(f"Saved {len(examples)} examples to {output_file}")
        
        # Print statistics
        logger.info(f"Dataset statistics:")
        logger.info(f"  Total examples: {len(examples)}")
        logger.info(f"  File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
        
        return str(output_file)

class AzureStorageUploader:
    """Handle Azure Storage uploads for training data"""
    
    def __init__(self, connection_string: Optional[str] = None, 
                 account_name: Optional[str] = None, 
                 account_key: Optional[str] = None):
        
        if not AZURE_STORAGE_AVAILABLE:
            raise ImportError("Azure Storage SDK not available. Install with: pip install azure-storage-blob")
        
        # Initialize blob service client
        if connection_string:
            self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        elif account_name and account_key:
            account_url = f"https://{account_name}.blob.core.windows.net"
            self.blob_service_client = BlobServiceClient(account_url=account_url, credential=account_key)
        elif account_name:
            # Use default credentials
            credential = DefaultAzureCredential()
            account_url = f"https://{account_name}.blob.core.windows.net"
            self.blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
        else:
            raise ValueError("Must provide either connection_string or account_name")
        
        self.container_name = "training-data"
        self._ensure_container_exists()
    
    def _ensure_container_exists(self):
        """Create container if it doesn't exist"""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            container_client.get_container_properties()
        except:
            try:
                self.blob_service_client.create_container(self.container_name)
                logger.info(f"Created container: {self.container_name}")
            except:
                logger.warning(f"Could not create/verify container: {self.container_name}")
    
    def upload_file(self, local_file_path: str, blob_name: Optional[str] = None) -> str:
        """Upload file to Azure Storage"""
        local_path = Path(local_file_path)
        
        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {local_file_path}")
        
        if blob_name is None:
            timestamp = int(time.time())
            blob_name = f"security_training_{timestamp}_{local_path.name}"
        
        logger.info(f"Uploading {local_file_path} to Azure Storage...")
        
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        
        with open(local_path, 'rb') as data:
            blob_client.upload_blob(data, overwrite=True)
        
        blob_url = f"https://{self.blob_service_client.account_name}.blob.core.windows.net/{self.container_name}/{blob_name}"
        
        logger.info(f"✅ Upload successful: {blob_url}")
        return blob_url

def main():
    parser = argparse.ArgumentParser(description="Generate Security Training Data")
    parser.add_argument("--output", default="security_training.jsonl", help="Output file path")
    parser.add_argument("--setup", action="store_true", help="Download/setup data sources")
    parser.add_argument("--data-path", default="./security_data", help="Base path for data sources")
    
    # Azure Storage options
    storage_group = parser.add_argument_group("Azure Storage Upload (Optional)")
    storage_group.add_argument("--upload-to-storage", action="store_true", help="Upload to Azure Storage")
    storage_group.add_argument("--storage-connection-string", help="Azure Storage connection string")
    storage_group.add_argument("--storage-account-name", help="Azure Storage account name")
    storage_group.add_argument("--storage-account-key", help="Azure Storage account key")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = SecurityTrainingDataGenerator(args.data_path)
    
    # Setup data sources if requested
    if args.setup:
        generator.setup_data_sources()
    
    # Generate training dataset
    training_examples = generator.generate_complete_dataset()
    
    # Save locally
    local_file = generator.save_dataset(training_examples, args.output)
    
    # Upload to Azure Storage if requested
    if args.upload_to_storage:
        if not any([args.storage_connection_string, args.storage_account_name]):
            logger.error("Azure Storage credentials required for upload")
            logger.info("Provide --storage-connection-string OR --storage-account-name")
            return
        
        try:
            uploader = AzureStorageUploader(
                connection_string=args.storage_connection_string,
                account_name=args.storage_account_name,
                account_key=args.storage_account_key
            )
            
            blob_url = uploader.upload_file(local_file)
            logger.info(f"🚀 Training data available at: {blob_url}")
            
        except Exception as e:
            logger.error(f"❌ Azure Storage upload failed: {e}")
            logger.info(f"💾 Training data saved locally: {local_file}")
    
    logger.info(f"""
🎉 TRAINING DATA GENERATION COMPLETE!

📁 Local file: {local_file}
📊 Examples: {len(training_examples)}
📝 Format: Azure OpenAI fine-tuning (messages format)

🚀 NEXT STEPS:
1. For AI Foundry: Upload {local_file} directly in the UI
2. For Azure ML: Use the Azure ML pipeline script
3. For local testing: Review the examples in the file

💡 TIP: The data is in the correct format for o4-mini fine-tuning in AI Foundry!
    """)

if __name__ == "__main__":
    main()