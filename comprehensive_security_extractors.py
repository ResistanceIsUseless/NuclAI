#!/usr/bin/env python3
"""
Comprehensive Security Data Extractors
Extracts training data from multiple security-focused repositories and documentation
"""

import os
import re
import json
import yaml
import requests
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union, Set
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging
import subprocess
import time
from urllib.parse import urljoin, urlparse
import markdown
from bs4 import BeautifulSoup
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the base classes from the previous framework
from enum import Enum

class SecurityDomain(Enum):
    NETWORK_SCANNING = "network_scanning"
    VULNERABILITY_ASSESSMENT = "vulnerability_assessment"
    PROTOCOL_ANALYSIS = "protocol_analysis"
    EXPLOIT_DEVELOPMENT = "exploit_development"
    THREAT_INTELLIGENCE = "threat_intelligence"
    COMPLIANCE = "compliance"
    WEB_APPLICATION_SECURITY = "web_application_security"
    PROXY_SECURITY = "proxy_security"
    BUG_BOUNTY = "bug_bounty"

class ReasoningType(Enum):
    SYNTHESIS = "synthesis"
    ADAPTATION = "adaptation"
    ANALYSIS = "analysis"
    DECISION_MAKING = "decision"
    TROUBLESHOOTING = "debug"
    PLANNING = "planning"
    EXPLOITATION = "exploitation"
    MITIGATION = "mitigation"

@dataclass
class SecurityConcept:
    name: str
    domain: SecurityDomain
    technique: str
    purpose: str
    implementation_approach: str
    prerequisites: List[str]
    indicators: List[str]
    related_vulnerabilities: List[str]
    compliance_frameworks: List[str] = None
    threat_actors: List[str] = None
    mitigations: List[str] = None
    severity: str = "medium"
    references: List[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass 
class TrainingExample:
    instruction: str
    input: str
    output: str
    reasoning_type: ReasoningType
    security_domain: SecurityDomain
    concepts_used: List[str]
    difficulty_level: str
    source_tool: str
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['reasoning_type'] = self.reasoning_type.value
        result['security_domain'] = self.security_domain.value
        return result

class SecurityDataExtractor(ABC):
    def __init__(self, source_path: str):
        self.source_path = Path(source_path)
        self.concepts = []
        
    @abstractmethod
    def extract_concepts(self) -> List[SecurityConcept]:
        pass
        
    @abstractmethod
    def generate_reasoning_examples(self, concepts: List[SecurityConcept]) -> List[TrainingExample]:
        pass

class NucleiTemplatesExtractor(SecurityDataExtractor):
    """Extract training data from ProjectDiscovery Nuclei Templates"""
    
    def __init__(self, source_path: str):
        super().__init__(source_path)
        self.repo_url = "https://github.com/projectdiscovery/nuclei-templates.git"
    
    def clone_or_update_repo(self):
        """Clone or update the Nuclei templates repository"""
        if not self.source_path.exists():
            logger.info(f"Cloning Nuclei templates to {self.source_path}")
            subprocess.run(["git", "clone", self.repo_url, str(self.source_path)], check=True)
        else:
            logger.info(f"Updating Nuclei templates at {self.source_path}")
            subprocess.run(["git", "pull"], cwd=self.source_path, check=True)
    
    def extract_concepts(self) -> List[SecurityConcept]:
        """Extract concepts from Nuclei YAML templates"""
        concepts = []
        
        if not self.source_path.exists():
            logger.error(f"Nuclei templates path not found: {self.source_path}")
            return concepts
            
        # Find all YAML template files, excluding test files
        yaml_files = []
        for pattern in ["*.yaml", "*.yml"]:
            yaml_files.extend(self.source_path.rglob(pattern))
        
        # Filter out test and example files
        yaml_files = [f for f in yaml_files if not any(exclude in str(f).lower() 
                     for exclude in ['test', 'example', '.github', 'workflows'])]
        
        logger.info(f"Processing {len(yaml_files)} Nuclei templates")
        
        for yaml_file in yaml_files[:1000]:  # Limit for performance
            try:
                with open(yaml_file, 'r', encoding='utf-8', errors='ignore') as f:
                    template = yaml.safe_load(f)
                
                if not isinstance(template, dict) or 'info' not in template:
                    continue
                    
                concept = self._extract_nuclei_concept(template, yaml_file)
                if concept:
                    concepts.append(concept)
                    
            except Exception as e:
                logger.debug(f"Failed to process {yaml_file}: {e}")
                
        logger.info(f"Extracted {len(concepts)} concepts from Nuclei templates")
        return concepts
    
    def _extract_nuclei_concept(self, template: Dict, file_path: Path) -> Optional[SecurityConcept]:
        """Extract security concept from Nuclei template"""
        info = template.get('info', {})
        
        name = info.get('name', file_path.stem)
        description = info.get('description', '')
        severity = info.get('severity', 'info')
        tags = info.get('tags', [])
        if isinstance(tags, str):
            tags = [tags]
        
        # Determine domain and technique
        domain = self._categorize_nuclei_domain(tags, template, file_path)
        technique = self._extract_nuclei_technique(template, tags)
        
        # Extract CVEs and references
        cves = []
        references = []
        classification = info.get('classification', {})
        
        if 'cve-id' in classification:
            cve_data = classification['cve-id']
            cves = cve_data if isinstance(cve_data, list) else [cve_data]
        
        if 'reference' in info:
            ref_data = info['reference']
            references = ref_data if isinstance(ref_data, list) else [ref_data]
        
        # Extract metadata
        metadata = info.get('metadata', {})
        
        return SecurityConcept(
            name=name,
            domain=domain,
            technique=technique,
            purpose=description,
            implementation_approach=self._extract_nuclei_implementation(template),
            prerequisites=self._extract_nuclei_prerequisites(template),
            indicators=self._extract_nuclei_indicators(template),
            related_vulnerabilities=cves,
            compliance_frameworks=[],
            severity=severity,
            references=references,
            mitigations=info.get('remediation', '').split(';') if info.get('remediation') else []
        )
    
    def _categorize_nuclei_domain(self, tags: List[str], template: Dict, file_path: Path) -> SecurityDomain:
        """Categorize template into security domain"""
        tag_str = ' '.join(tags).lower()
        path_str = str(file_path).lower()
        
        if any(word in tag_str or word in path_str for word in ['cve', 'vuln', 'rce', 'sqli', 'xss']):
            return SecurityDomain.VULNERABILITY_ASSESSMENT
        elif any(word in tag_str or word in path_str for word in ['network', 'port', 'service', 'discovery']):
            return SecurityDomain.NETWORK_SCANNING
        elif any(word in tag_str or word in path_str for word in ['http', 'web', 'application']):
            return SecurityDomain.WEB_APPLICATION_SECURITY
        elif any(word in tag_str or word_str in path_str for word in ['ssl', 'tls', 'protocol']):
            return SecurityDomain.PROTOCOL_ANALYSIS
        else:
            return SecurityDomain.VULNERABILITY_ASSESSMENT
    
    def _extract_nuclei_technique(self, template: Dict, tags: List[str]) -> str:
        """Extract the testing technique"""
        tag_str = ' '.join(tags).lower()
        
        if 'http' in template:
            if any(word in tag_str for word in ['sqli', 'sql']):
                return "sql_injection_testing"
            elif any(word in tag_str for word in ['xss', 'cross-site']):
                return "xss_vulnerability_testing"
            elif any(word in tag_str for word in ['rce', 'command']):
                return "remote_code_execution_testing"
            elif any(word in tag_str for word in ['lfi', 'file']):
                return "file_inclusion_testing"
            else:
                return "http_vulnerability_scanning"
        elif 'network' in template:
            return "network_service_testing"
        elif 'dns' in template:
            return "dns_security_testing"
        elif any(word in tag_str for word in ['ssl', 'tls']):
            return "ssl_tls_testing"
        else:
            return "generic_security_testing"
    
    def _extract_nuclei_implementation(self, template: Dict) -> str:
        """Extract implementation approach"""
        approaches = []
        
        if 'http' in template:
            http_config = template['http']
            if isinstance(http_config, list) and len(http_config) > 1:
                approaches.append("multi_step_requests")
            else:
                approaches.append("http_requests")
                
        if 'network' in template:
            approaches.append("network_protocols")
            
        if any(section in template for section in ['matchers', 'extractors']):
            approaches.append("pattern_matching")
            
        return ','.join(approaches) if approaches else "template_based"
    
    def _extract_nuclei_prerequisites(self, template: Dict) -> List[str]:
        """Extract prerequisites"""
        prereqs = []
        
        if 'http' in template:
            prereqs.append("http_service_accessible")
            
        if 'network' in template:
            prereqs.append("network_connectivity")
            
        # Check for authentication requirements
        if any('auth' in str(template).lower() for _ in [1]):
            prereqs.append("authentication_required")
            
        return prereqs
    
    def _extract_nuclei_indicators(self, template: Dict) -> List[str]:
        """Extract detection indicators"""
        indicators = []
        
        def extract_from_matchers(section):
            if isinstance(section, list):
                for item in section:
                    if isinstance(item, dict) and 'matchers' in item:
                        for matcher in item['matchers']:
                            if isinstance(matcher, dict):
                                if 'words' in matcher:
                                    words = matcher['words']
                                    if isinstance(words, list):
                                        indicators.extend(words[:3])
                                if 'regex' in matcher:
                                    regex_patterns = matcher['regex']
                                    if isinstance(regex_patterns, list):
                                        indicators.extend(regex_patterns[:2])
        
        for section_name in ['http', 'network', 'dns']:
            if section_name in template:
                extract_from_matchers(template[section_name])
                
        return indicators[:10]  # Limit to avoid noise
    
    def generate_reasoning_examples(self, concepts: List[SecurityConcept]) -> List[TrainingExample]:
        """Generate Nuclei-specific reasoning examples"""
        examples = []
        
        # Group concepts by technique for better reasoning
        technique_groups = {}
        for concept in concepts:
            if concept.technique not in technique_groups:
                technique_groups[concept.technique] = []
            technique_groups[concept.technique].append(concept)
        
        # Generate technique combination examples
        techniques = list(technique_groups.keys())
        for i, tech1 in enumerate(techniques[:10]):
            for tech2 in techniques[i+1:i+3]:
                concepts1 = technique_groups[tech1][:2]
                concepts2 = technique_groups[tech2][:2]
                
                for c1 in concepts1:
                    for c2 in concepts2:
                        examples.append(TrainingExample(
                            instruction=f"Design a comprehensive web application security test combining {tech1.replace('_', ' ')} and {tech2.replace('_', ' ')}",
                            input=f"Target application may be vulnerable to issues detected by both techniques. Need systematic approach.",
                            output=f"A comprehensive test should sequence {tech1} and {tech2} strategically. Start with {tech1} because {c1.purpose[:80]}. This technique looks for indicators like {', '.join(c1.indicators[:2])} and requires {', '.join(c1.prerequisites)}. Follow with {tech2} to {c2.purpose[:80]}. The second technique validates through {', '.join(c2.indicators[:2])} and may reveal additional attack vectors. Cross-correlate findings to identify chained vulnerabilities and reduce false positives.",
                            reasoning_type=ReasoningType.SYNTHESIS,
                            security_domain=c1.domain,
                            concepts_used=[c1.name, c2.name],
                            difficulty_level="intermediate",
                            source_tool="nuclei"
                        ))
        
        # Generate adaptation examples
        for concept in concepts[:20]:
            if concept.related_vulnerabilities:
                examples.append(TrainingExample(
                    instruction=f"Adapt the {concept.technique.replace('_', ' ')} approach for a custom enterprise application",
                    input=f"Standard template: {concept.name}. Target: Custom enterprise app with modified architecture.",
                    output=f"To adapt {concept.technique} for a custom environment: 1) Analyze the core vulnerability pattern that {concept.purpose[:60]} targets, specifically {', '.join(concept.related_vulnerabilities)}, 2) Map standard indicators {', '.join(concept.indicators[:2])} to custom application responses, 3) Modify request patterns in the {concept.implementation_approach} approach to match custom endpoints, 4) Adjust detection logic for environment-specific error messages or behaviors, 5) Validate that prerequisites {', '.join(concept.prerequisites)} are met in the custom context.",
                    reasoning_type=ReasoningType.ADAPTATION,
                    security_domain=concept.domain,
                    concepts_used=[concept.name],
                    difficulty_level="advanced",
                    source_tool="nuclei"
                ))
        
        # Generate severity-based decision making examples
        severity_groups = {}
        for concept in concepts:
            if concept.severity not in severity_groups:
                severity_groups[concept.severity] = []
            severity_groups[concept.severity].append(concept)
        
        for severity, severity_concepts in severity_groups.items():
            if len(severity_concepts) >= 2:
                c1, c2 = severity_concepts[:2]
                examples.append(TrainingExample(
                    instruction=f"Prioritize security testing when multiple {severity} severity vulnerabilities are possible",
                    input=f"Limited testing time. Both {c1.technique} and {c2.technique} could reveal {severity} issues.",
                    output=f"When prioritizing {severity} severity testing with time constraints: 1) Assess which technique has higher success probability - {c1.technique} requires {', '.join(c1.prerequisites)} while {c2.technique} needs {', '.join(c2.prerequisites)}, 2) Consider attack surface - {c1.purpose[:50]} versus {c2.purpose[:50]}, 3) Evaluate detection confidence - {c1.technique} looks for {', '.join(c1.indicators[:2])} while {c2.technique} identifies {', '.join(c2.indicators[:2])}, 4) Choose the technique with clearer prerequisites and indicators first, then proceed based on initial results.",
                    reasoning_type=ReasoningType.DECISION_MAKING,
                    security_domain=c1.domain,
                    concepts_used=[c1.name, c2.name],
                    difficulty_level="intermediate",
                    source_tool="nuclei"
                ))
        
        return examples[:50]  # Limit output

class BugBountyReportsExtractor(SecurityDataExtractor):
    """Extract training data from disclosed bug bounty reports"""
    
    def __init__(self, source_path: str):
        super().__init__(source_path)
        self.repo_url = "https://github.com/marcotuliocnd/bugbounty-disclosed-reports.git"
    
    def clone_or_update_repo(self):
        """Clone or update the bug bounty reports repository"""
        if not self.source_path.exists():
            logger.info(f"Cloning bug bounty reports to {self.source_path}")
            subprocess.run(["git", "clone", self.repo_url, str(self.source_path)], check=True)
        else:
            logger.info(f"Updating bug bounty reports at {self.source_path}")
            subprocess.run(["git", "pull"], cwd=self.source_path, check=True)
    
    def extract_concepts(self) -> List[SecurityConcept]:
        """Extract concepts from bug bounty reports"""
        concepts = []
        
        if not self.source_path.exists():
            logger.error(f"Bug bounty reports path not found: {self.source_path}")
            return concepts
        
        # Find markdown files containing reports
        md_files = list(self.source_path.rglob("*.md"))
        logger.info(f"Processing {len(md_files)} bug bounty report files")
        
        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                report_concepts = self._extract_report_concepts(content, md_file)
                concepts.extend(report_concepts)
                
            except Exception as e:
                logger.debug(f"Failed to process {md_file}: {e}")
        
        logger.info(f"Extracted {len(concepts)} concepts from bug bounty reports")
        return concepts
    
    def _extract_report_concepts(self, content: str, file_path: Path) -> List[SecurityConcept]:
        """Extract concepts from individual bug bounty report"""
        concepts = []
        
        # Parse markdown to extract structured information
        lines = content.split('\n')
        current_report = {}
        
        for line in lines:
            line = line.strip()
            
            # Look for report headers and key information
            if line.startswith('#'):
                if current_report and 'title' in current_report:
                    concept = self._create_bug_bounty_concept(current_report, file_path)
                    if concept:
                        concepts.append(concept)
                current_report = {'title': line.strip('#').strip()}
            
            # Extract key fields
            elif ':' in line and current_report:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key in ['severity', 'bounty', 'vulnerability', 'impact', 'poc', 'steps']:
                    current_report[key] = value
            
            # Extract vulnerability descriptions
            elif line and not line.startswith('#') and current_report:
                if 'description' not in current_report:
                    current_report['description'] = line
                else:
                    current_report['description'] += ' ' + line
        
        # Process final report
        if current_report and 'title' in current_report:
            concept = self._create_bug_bounty_concept(current_report, file_path)
            if concept:
                concepts.append(concept)
        
        return concepts
    
    def _create_bug_bounty_concept(self, report: Dict, file_path: Path) -> Optional[SecurityConcept]:
        """Create security concept from bug bounty report data"""
        if 'title' not in report:
            return None
        
        title = report['title']
        description = report.get('description', '')[:200]  # Limit description
        
        # Categorize vulnerability type
        vuln_type = self._categorize_vulnerability(title, description)
        domain = self._map_vuln_to_domain(vuln_type)
        
        # Extract severity
        severity = report.get('severity', 'medium').lower()
        
        # Extract indicators and techniques
        indicators = self._extract_bug_bounty_indicators(description)
        technique = self._extract_bug_bounty_technique(vuln_type, description)
        
        return SecurityConcept(
            name=f"bugbounty_{hashlib.md5(title.encode()).hexdigest()[:8]}",
            domain=domain,
            technique=technique,
            purpose=f"Real-world exploitation: {description[:100]}",
            implementation_approach="manual_testing",
            prerequisites=["target_access", "manual_testing_skills"],
            indicators=indicators,
            related_vulnerabilities=[vuln_type],
            severity=severity,
            references=[str(file_path)]
        )
    
    def _categorize_vulnerability(self, title: str, description: str) -> str:
        """Categorize the type of vulnerability"""
        text = (title + ' ' + description).lower()
        
        vuln_patterns = {
            'sql_injection': ['sql', 'sqli', 'injection', 'database'],
            'xss': ['xss', 'cross-site', 'script', 'javascript'],
            'rce': ['rce', 'remote code', 'command injection', 'execution'],
            'ssrf': ['ssrf', 'server-side request', 'internal'],
            'idor': ['idor', 'insecure direct', 'object reference'],
            'csrf': ['csrf', 'cross-site request'],
            'lfi': ['lfi', 'local file', 'path traversal'],
            'xxe': ['xxe', 'xml external', 'entity'],
            'authorization_bypass': ['bypass', 'privilege', 'escalation'],
            'information_disclosure': ['disclosure', 'leak', 'exposure']
        }
        
        for vuln_type, keywords in vuln_patterns.items():
            if any(keyword in text for keyword in keywords):
                return vuln_type
        
        return 'unknown_vulnerability'
    
    def _map_vuln_to_domain(self, vuln_type: str) -> SecurityDomain:
        """Map vulnerability type to security domain"""
        web_vulns = ['sql_injection', 'xss', 'csrf', 'xxe', 'lfi']
        if vuln_type in web_vulns:
            return SecurityDomain.WEB_APPLICATION_SECURITY
        else:
            return SecurityDomain.BUG_BOUNTY
    
    def _extract_bug_bounty_indicators(self, description: str) -> List[str]:
        """Extract indicators from bug bounty description"""
        indicators = []
        
        # Look for specific patterns, endpoints, parameters
        url_pattern = re.findall(r'https?://[^\s]+', description)
        indicators.extend(url_pattern[:3])
        
        # Look for parameter names
        param_pattern = re.findall(r'[?&]([^=&\s]+)=', description)
        indicators.extend(param_pattern[:3])
        
        # Look for error messages
        error_pattern = re.findall(r'"([^"]*error[^"]*)"', description, re.IGNORECASE)
        indicators.extend(error_pattern[:2])
        
        return indicators
    
    def _extract_bug_bounty_technique(self, vuln_type: str, description: str) -> str:
        """Extract technique used in bug bounty report"""
        if 'automation' in description.lower() or 'tool' in description.lower():
            return f"automated_{vuln_type}_detection"
        else:
            return f"manual_{vuln_type}_testing"
    
    def generate_reasoning_examples(self, concepts: List[SecurityConcept]) -> List[TrainingExample]:
        """Generate bug bounty specific reasoning examples"""
        examples = []
        
        # Group by vulnerability type
        vuln_groups = {}
        for concept in concepts:
            for vuln in concept.related_vulnerabilities:
                if vuln not in vuln_groups:
                    vuln_groups[vuln] = []
                vuln_groups[vuln].append(concept)
        
        # Generate real-world exploitation examples
        for vuln_type, vuln_concepts in vuln_groups.items():
            if len(vuln_concepts) >= 2:
                c1, c2 = vuln_concepts[:2]
                examples.append(TrainingExample(
                    instruction=f"Develop a comprehensive {vuln_type.replace('_', ' ')} testing methodology based on real-world findings",
                    input=f"Multiple disclosed reports show {vuln_type} vulnerabilities. Need systematic approach.",
                    output=f"Based on real-world {vuln_type} disclosures, an effective methodology should: 1) Start with {c1.technique} as demonstrated in actual reports, 2) Look for indicators like {', '.join(c1.indicators[:2])} which have proven successful, 3) Apply {c2.technique} for additional coverage, focusing on {', '.join(c2.indicators[:2])}, 4) Prioritize testing areas where bug bounty hunters have found success, 5) Document findings with clear impact demonstration as shown in disclosed reports.",
                    reasoning_type=ReasoningType.EXPLOITATION,
                    security_domain=SecurityDomain.BUG_BOUNTY,
                    concepts_used=[c1.name, c2.name],
                    difficulty_level="advanced",
                    source_tool="bugbounty"
                ))
        
        return examples[:20]

class RFCExtractor(SecurityDataExtractor):
    """Extract training data from Internet RFCs"""
    
    def __init__(self, source_path: str):
        super().__init__(source_path)
        self.rfc_urls = [
            "https://www.ietf.org/rfc/rfc9110.txt",  # HTTP Semantics
            "https://www.ietf.org/rfc/rfc2068.txt",  # HTTP/1.1
            "https://www.ietf.org/rfc/rfc7231.txt",  # HTTP/1.1 Semantics
            "https://www.ietf.org/rfc/rfc7232.txt",  # HTTP/1.1 Conditional Requests
            "https://www.ietf.org/rfc/rfc7233.txt",  # HTTP/1.1 Range Requests
            "https://www.ietf.org/rfc/rfc7234.txt",  # HTTP/1.1 Caching
            "https://www.ietf.org/rfc/rfc7235.txt",  # HTTP/1.1 Authentication
        ]
    
    def download_rfcs(self):
        """Download RFC documents"""
        self.source_path.mkdir(exist_ok=True)
        
        for url in self.rfc_urls:
            rfc_name = url.split('/')[-1]
            rfc_path = self.source_path / rfc_name
            
            if not rfc_path.exists():
                logger.info(f"Downloading {rfc_name}")
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    with open(rfc_path, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    time.sleep(1)  # Be respectful
                except Exception as e:
                    logger.warning(f"Failed to download {url}: {e}")
    
    def extract_concepts(self) -> List[SecurityConcept]:
        """Extract concepts from RFC documents"""
        concepts = []
        
        if not self.source_path.exists():
            self.source_path.mkdir(exist_ok=True)
        
        # Download RFCs if needed
        self.download_rfcs()
        
        rfc_files = list(self.source_path.glob("*.txt"))
        logger.info(f"Processing {len(rfc_files)} RFC documents")
        
        for rfc_file in rfc_files:
            try:
                with open(rfc_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                rfc_concepts = self._extract_rfc_concepts(content, rfc_file)
                concepts.extend(rfc_concepts)
                
            except Exception as e:
                logger.debug(f"Failed to process {rfc_file}: {e}")
        
        logger.info(f"Extracted {len(concepts)} concepts from RFCs")
        return concepts
    
    def _extract_rfc_concepts(self, content: str, file_path: Path) -> List[SecurityConcept]:
        """Extract security concepts from RFC content"""
        concepts = []
        
        # Extract security considerations section
        security_section = self._extract_security_section(content)
        
        # Extract protocol mechanisms that have security implications
        mechanisms = self._extract_protocol_mechanisms(content)
        
        for mechanism in mechanisms:
            concept = SecurityConcept(
                name=f"rfc_{mechanism['name'].lower().replace(' ', '_')}",
                domain=SecurityDomain.PROTOCOL_ANALYSIS,
                technique="protocol_security_analysis",
                purpose=mechanism['purpose'],
                implementation_approach="protocol_implementation",
                prerequisites=["protocol_knowledge", "network_access"],
                indicators=mechanism['indicators'],
                related_vulnerabilities=mechanism['vulnerabilities'],
                references=[str(file_path)]
            )
            concepts.append(concept)
        
        return concepts
    
    def _extract_security_section(self, content: str) -> str:
        """Extract security considerations section from RFC"""
        # Look for security considerations section
        security_pattern = re.search(
            r'(\d+\.?\s*Security Considerations.*?)(?=\d+\.?\s*[A-Z]|\n\s*References|\Z)',
            content, re.DOTALL | re.IGNORECASE
        )
        
        return security_pattern.group(1) if security_pattern else ""
    
    def _extract_protocol_mechanisms(self, content: str) -> List[Dict]:
        """Extract protocol mechanisms with security implications"""
        mechanisms = []
        
        # Look for authentication mechanisms
        auth_pattern = re.findall(
            r'(authentication|authorization)[^.]*\.([^.]*\.){0,2}',
            content, re.IGNORECASE
        )
        
        if auth_pattern:
            mechanisms.append({
                'name': 'HTTP Authentication',
                'purpose': 'Provide authentication mechanisms for HTTP requests',
                'indicators': ['WWW-Authenticate', 'Authorization', '401', '403'],
                'vulnerabilities': ['credential_exposure', 'replay_attacks']
            })
        
        # Look for caching mechanisms
        cache_pattern = re.findall(
            r'(cache|caching)[^.]*\.([^.]*\.){0,2}',
            content, re.IGNORECASE
        )
        
        if cache_pattern:
            mechanisms.append({
                'name': 'HTTP Caching',
                'purpose': 'Control caching behavior of HTTP responses',
                'indicators': ['Cache-Control', 'ETag', 'Last-Modified', 'Expires'],
                'vulnerabilities': ['cache_poisoning', 'sensitive_data_caching']
            })
        
        # Look for header mechanisms
        header_pattern = re.findall(
            r'([A-Z][a-z]+-[A-Z][a-z]+)\s*header',
            content
        )
        
        for header in set(header_pattern[:10]):  # Limit and deduplicate
            mechanisms.append({
                'name': f'{header} Header Security',
                'purpose': f'Security implications of {header} header usage',
                'indicators': [header, 'header manipulation', 'header injection'],
                'vulnerabilities': ['header_injection', 'response_splitting']
            })
        
        return mechanisms
    
    def generate_reasoning_examples(self, concepts: List[SecurityConcept]) -> List[TrainingExample]:
        """Generate RFC-based reasoning examples"""
        examples = []
        
        # Generate protocol security analysis examples
        for concept in concepts[:15]:
            examples.append(TrainingExample(
                instruction=f"Analyze the security implications of {concept.name.replace('_', ' ')} in modern web applications",
                input=f"Protocol mechanism: {concept.purpose}",
                output=f"The {concept.name.replace('_', ' ')} mechanism has several security implications: 1) {concept.purpose.lower()}, which can be vulnerable to {', '.join(concept.related_vulnerabilities)}, 2) Indicators to monitor include {', '.join(concept.indicators[:3])}, 3) Implementation requires careful consideration of {', '.join(concept.prerequisites)}, 4) Security controls should focus on preventing {', '.join(concept.related_vulnerabilities)} through proper validation and sanitization, 5) Monitoring should detect anomalous patterns in {', '.join(concept.indicators[:2])} usage.",
                reasoning_type=ReasoningType.ANALYSIS,
                security_domain=SecurityDomain.PROTOCOL_ANALYSIS,
                concepts_used=[concept.name],
                difficulty_level="expert",
                source_tool="rfc"
            ))
        
        return examples

class MDNDocsExtractor(SecurityDataExtractor):
    """Extract training data from MDN documentation"""
    
    def __init__(self, source_path: str):
        super().__init__(source_path)
        self.mdn_urls = [
            "https://developer.mozilla.org/en-US/docs/Web/HTTP/Proxy_servers_and_tunneling",
            "https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers",
            "https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods",
            "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status",
            "https://developer.mozilla.org/en-US/docs/Web/Security"
        ]
    
    def download_mdn_docs(self):
        """Download MDN documentation"""
        self.source_path.mkdir(exist_ok=True)
        
        for url in self.mdn_urls:
            doc_name = url.split('/')[-1] + ".html"
            doc_path = self.source_path / doc_name
            
            if not doc_path.exists():
                logger.info(f"Downloading {doc_name}")
                try:
                    headers = {'User-Agent': 'Mozilla/5.0 (Educational Research Bot)'}
                    response = requests.get(url, headers=headers)
                    response.raise_for_status()
                    with open(doc_path, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    time.sleep(2)  # Be respectful
                except Exception as e:
                    logger.warning(f"Failed to download {url}: {e}")
    
    def extract_concepts(self) -> List[SecurityConcept]:
        """Extract concepts from MDN documentation"""
        concepts = []
        
        if not self.source_path.exists():
            self.source_path.mkdir(exist_ok=True)
        
        # Download docs if needed
        self.download_mdn_docs()
        
        html_files = list(self.source_path.glob("*.html"))
        logger.info(f"Processing {len(html_files)} MDN documents")
        
        for html_file in html_files:
            try:
                with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                mdn_concepts = self._extract_mdn_concepts(content, html_file)
                concepts.extend(mdn_concepts)
                
            except Exception as e:
                logger.debug(f"Failed to process {html_file}: {e}")
        
        logger.info(f"Extracted {len(concepts)} concepts from MDN docs")
        return concepts
    
    def _extract_mdn_concepts(self, content: str, file_path: Path) -> List[SecurityConcept]:
        """Extract security concepts from MDN HTML content"""
        concepts = []
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract proxy-related security concepts
        if 'proxy' in file_path.name.lower():
            concepts.extend(self._extract_proxy_concepts(soup))
        
        # Extract header security concepts
        if 'headers' in file_path.name.lower():
            concepts.extend(self._extract_header_security_concepts(soup))
        
        # Extract general security concepts
        security_sections = soup.find_all(['section', 'div'], 
                                        string=re.compile(r'security', re.I))
        for section in security_sections:
            concepts.extend(self._extract_general_security_concepts(section))
        
        return concepts
    
    def _extract_proxy_concepts(self, soup: BeautifulSoup) -> List[SecurityConcept]:
        """Extract proxy-related security concepts"""
        concepts = []
        
        # Look for proxy-related security information
        proxy_concepts = [
            {
                'name': 'HTTP Proxy Security',
                'purpose': 'Understanding security implications of HTTP proxies and tunneling',
                'vulnerabilities': ['proxy_bypass', 'tunnel_abuse', 'proxy_injection'],
                'indicators': ['CONNECT', 'Via', 'X-Forwarded-For', 'Proxy-Authorization']
            },
            {
                'name': 'Proxy Authentication',
                'purpose': 'Secure authentication mechanisms for proxy servers',
                'vulnerabilities': ['credential_leakage', 'authentication_bypass'],
                'indicators': ['Proxy-Authenticate', 'Proxy-Authorization', '407']
            }
        ]
        
        for concept_data in proxy_concepts:
            concept = SecurityConcept(
                name=concept_data['name'].lower().replace(' ', '_'),
                domain=SecurityDomain.PROXY_SECURITY,
                technique="proxy_security_analysis",
                purpose=concept_data['purpose'],
                implementation_approach="proxy_configuration_analysis",
                prerequisites=["proxy_access", "network_monitoring"],
                indicators=concept_data['indicators'],
                related_vulnerabilities=concept_data['vulnerabilities'],
                references=[str(file_path)]
            )
            concepts.append(concept)
        
        return concepts
    
    def _extract_header_security_concepts(self, soup: BeautifulSoup) -> List[SecurityConcept]:
        """Extract HTTP header security concepts"""
        concepts = []
        
        # Security-related headers
        security_headers = [
            'Content-Security-Policy',
            'Strict-Transport-Security',
            'X-Content-Type-Options',
            'X-Frame-Options',
            'X-XSS-Protection'
        ]
        
        for header in security_headers:
            concept = SecurityConcept(
                name=f"{header.lower().replace('-', '_')}_security",
                domain=SecurityDomain.WEB_APPLICATION_SECURITY,
                technique="security_header_analysis",
                purpose=f"Security implementation and analysis of {header} header",
                implementation_approach="header_configuration",
                prerequisites=["web_server_access", "http_knowledge"],
                indicators=[header, "header misconfiguration", "missing security headers"],
                related_vulnerabilities=["header_injection", "clickjacking", "xss"],
                references=[str(file_path)]
            )
            concepts.append(concept)
        
        return concepts
    
    def _extract_general_security_concepts(self, section) -> List[SecurityConcept]:
        """Extract general security concepts from sections"""
        concepts = []
        
        # Extract text content for analysis
        text = section.get_text() if section else ""
        
        if len(text) > 100:  # Only process substantial content
            concept = SecurityConcept(
                name=f"mdn_security_{hashlib.md5(text[:50].encode()).hexdigest()[:8]}",
                domain=SecurityDomain.WEB_APPLICATION_SECURITY,
                technique="web_security_analysis",
                purpose=text[:150],  # First 150 chars as purpose
                implementation_approach="web_standards_compliance",
                prerequisites=["web_development_knowledge"],
                indicators=["security headers", "secure configuration"],
                related_vulnerabilities=["web_vulnerabilities"],
                references=["mdn_documentation"]
            )
            concepts.append(concept)
        
        return concepts
    
    def generate_reasoning_examples(self, concepts: List[SecurityConcept]) -> List[TrainingExample]:
        """Generate MDN-based reasoning examples"""
        examples = []
        
        # Generate web security implementation examples
        for concept in concepts[:10]:
            examples.append(TrainingExample(
                instruction=f"Implement secure web application practices based on {concept.name.replace('_', ' ')}",
                input=f"Web application security requirement: {concept.purpose[:80]}",
                output=f"To implement {concept.name.replace('_', ' ')} securely: 1) Understand that {concept.purpose[:60]}, 2) Configure {concept.implementation_approach} to address {', '.join(concept.related_vulnerabilities)}, 3) Monitor for indicators such as {', '.join(concept.indicators[:2])}, 4) Ensure prerequisites {', '.join(concept.prerequisites)} are met, 5) Follow web standards and security best practices for proper implementation.",
                reasoning_type=ReasoningType.MITIGATION,
                security_domain=concept.domain,
                concepts_used=[concept.name],
                difficulty_level="intermediate",
                source_tool="mdn"
            ))
        
        return examples

class WeirdProxiesExtractor(SecurityDataExtractor):
    """Extract training data from weird proxies repository"""
    
    def __init__(self, source_path: str):
        super().__init__(source_path)
        self.repo_url = "https://github.com/GrrrDog/weird_proxies.git"
    
    def clone_or_update_repo(self):
        """Clone or update the weird proxies repository"""
        if not self.source_path.exists():
            logger.info(f"Cloning weird proxies to {self.source_path}")
            subprocess.run(["git", "clone", self.repo_url, str(self.source_path)], check=True)
        else:
            logger.info(f"Updating weird proxies at {self.source_path}")
            subprocess.run(["git", "pull"], cwd=self.source_path, check=True)
    
    def extract_concepts(self) -> List[SecurityConcept]:
        """Extract concepts from weird proxies data"""
        concepts = []
        
        if not self.source_path.exists():
            logger.error(f"Weird proxies path not found: {self.source_path}")
            return concepts
        
        # Find all relevant files
        files = list(self.source_path.rglob("*.md")) + list(self.source_path.rglob("*.txt"))
        logger.info(f"Processing {len(files)} weird proxies files")
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                proxy_concepts = self._extract_weird_proxy_concepts(content, file_path)
                concepts.extend(proxy_concepts)
                
            except Exception as e:
                logger.debug(f"Failed to process {file_path}: {e}")
        
        logger.info(f"Extracted {len(concepts)} concepts from weird proxies")
        return concepts
    
    def _extract_weird_proxy_concepts(self, content: str, file_path: Path) -> List[SecurityConcept]:
        """Extract proxy security concepts from weird proxies content"""
        concepts = []
        
        # Extract proxy behavior patterns
        proxy_behaviors = self._extract_proxy_behaviors(content)
        
        for behavior in proxy_behaviors:
            concept = SecurityConcept(
                name=f"weird_proxy_{behavior['name'].lower().replace(' ', '_')}",
                domain=SecurityDomain.PROXY_SECURITY,
                technique="proxy_behavior_analysis",
                purpose=behavior['description'],
                implementation_approach="proxy_testing",
                prerequisites=["proxy_access", "traffic_analysis_tools"],
                indicators=behavior['indicators'],
                related_vulnerabilities=behavior['vulnerabilities'],
                references=[str(file_path)]
            )
            concepts.append(concept)
        
        return concepts
    
    def _extract_proxy_behaviors(self, content: str) -> List[Dict]:
        """Extract proxy behavior patterns from content"""
        behaviors = []
        
        # Look for proxy-specific behaviors and configurations
        lines = content.split('\n')
        current_behavior = None
        
        for line in lines:
            line = line.strip()
            
            # Look for behavior descriptions
            if any(keyword in line.lower() for keyword in ['proxy', 'behavior', 'weird', 'unusual']):
                if current_behavior:
                    behaviors.append(current_behavior)
                
                current_behavior = {
                    'name': line[:50],
                    'description': line,
                    'indicators': [],
                    'vulnerabilities': ['proxy_abuse', 'traffic_manipulation']
                }
            
            # Extract technical details
            elif current_behavior and any(char in line for char in [':', '->', '=>']):
                current_behavior['indicators'].append(line[:30])
        
        if current_behavior:
            behaviors.append(current_behavior)
        
        # Add some default proxy security concepts
        if not behaviors:
            behaviors = [
                {
                    'name': 'Proxy Header Manipulation',
                    'description': 'Unusual proxy header handling and manipulation techniques',
                    'indicators': ['X-Forwarded-For', 'Via', 'X-Real-IP'],
                    'vulnerabilities': ['header_injection', 'ip_spoofing']
                },
                {
                    'name': 'Proxy Chain Analysis',
                    'description': 'Understanding complex proxy chain behaviors',
                    'indicators': ['multiple Via headers', 'proxy loops'],
                    'vulnerabilities': ['proxy_chaining_abuse', 'anonymity_bypass']
                }
            ]
        
        return behaviors
    
    def generate_reasoning_examples(self, concepts: List[SecurityConcept]) -> List[TrainingExample]:
        """Generate weird proxies reasoning examples"""
        examples = []
        
        # Generate proxy security analysis examples
        for concept in concepts[:10]:
            examples.append(TrainingExample(
                instruction=f"Analyze unusual proxy behavior: {concept.name.replace('_', ' ')}",
                input=f"Observed proxy behavior: {concept.purpose[:60]}",
                output=f"This unusual proxy behavior indicates: 1) {concept.purpose[:80]}, 2) Key indicators to look for include {', '.join(concept.indicators[:3])}, 3) Potential security implications include {', '.join(concept.related_vulnerabilities)}, 4) Analysis requires {', '.join(concept.prerequisites)}, 5) This behavior could be exploited for traffic manipulation or security bypass, requiring careful monitoring and controls.",
                reasoning_type=ReasoningType.ANALYSIS,
                security_domain=SecurityDomain.PROXY_SECURITY,
                concepts_used=[concept.name],
                difficulty_level="expert",
                source_tool="weird_proxies"
            ))
        
        return examples

class ComprehensiveSecurityTrainingGenerator:
    """Main class to orchestrate all extractors"""
    
    def __init__(self, base_path: str = "./security_data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        self.extractors = {
            'nuclei': NucleiTemplatesExtractor(self.base_path / "nuclei-templates"),
            'bugbounty': BugBountyReportsExtractor(self.base_path / "bugbounty-reports"),
            'rfc': RFCExtractor(self.base_path / "rfcs"),
            'mdn': MDNDocsExtractor(self.base_path / "mdn-docs"),
            'weird_proxies': WeirdProxiesExtractor(self.base_path / "weird-proxies")
        }
        
        self.all_concepts = []
        self.all_examples = []
    
    def setup_data_sources(self):
        """Setup and download all data sources"""
        logger.info("Setting up data sources...")
        
        # Clone/update repositories
        for name, extractor in self.extractors.items():
            if hasattr(extractor, 'clone_or_update_repo'):
                try:
                    extractor.clone_or_update_repo()
                except Exception as e:
                    logger.warning(f"Failed to setup {name}: {e}")
            elif hasattr(extractor, 'download_rfcs'):
                try:
                    extractor.download_rfcs()
                except Exception as e:
                    logger.warning(f"Failed to download RFCs: {e}")
            elif hasattr(extractor, 'download_mdn_docs'):
                try:
                    extractor.download_mdn_docs()
                except Exception as e:
                    logger.warning(f"Failed to download MDN docs: {e}")
    
    def generate_comprehensive_dataset(self) -> List[TrainingExample]:
        """Generate comprehensive training dataset from all sources"""
        logger.info("Generating comprehensive security training dataset...")
        
        # Extract concepts from all sources
        for name, extractor in self.extractors.items():
            try:
                logger.info(f"Processing {name}...")
                concepts = extractor.extract_concepts()
                self.all_concepts.extend(concepts)
                
                # Generate source-specific examples
                examples = extractor.generate_reasoning_examples(concepts)
                self.all_examples.extend(examples)
                
                logger.info(f"Extracted {len(concepts)} concepts and {len(examples)} examples from {name}")
                
            except Exception as e:
                logger.error(f"Failed to process {name}: {e}")
        
        # Generate cross-domain synthesis examples
        logger.info("Generating cross-domain synthesis examples...")
        cross_examples = self._generate_advanced_cross_domain_examples()
        self.all_examples.extend(cross_examples)
        
        logger.info(f"Total: {len(self.all_concepts)} concepts, {len(self.all_examples)} training examples")
        return self.all_examples
    
    def _generate_advanced_cross_domain_examples(self) -> List[TrainingExample]:
        """Generate advanced examples combining multiple domains and tools"""
        examples = []
        
        # Group concepts by domain
        domain_groups = {}
        for concept in self.all_concepts:
            if concept.domain not in domain_groups:
                domain_groups[concept.domain] = []
            domain_groups[concept.domain].append(concept)
        
        # Generate comprehensive security assessment examples
        if len(domain_groups) >= 2:
            domains = list(domain_groups.keys())
            for i, domain1 in enumerate(domains[:3]):
                for domain2 in domains[i+1:i+2]:
                    if domain_groups[domain1] and domain_groups[domain2]:
                        concept1 = domain_groups[domain1][0]
                        concept2 = domain_groups[domain2][0]
                        
                        examples.append(TrainingExample(
                            instruction=f"Design a comprehensive security assessment combining {domain1.value} and {domain2.value} approaches",
                            input=f"Target requires both {domain1.value} and {domain2.value} analysis for complete security coverage",
                            output=f"A comprehensive assessment integrating {domain1.value} and {domain2.value} should: 1) Begin with {concept1.technique} to {concept1.purpose[:50]}, establishing baseline security posture, 2) Apply {concept2.technique} to {concept2.purpose[:50]}, providing complementary analysis, 3) Cross-correlate findings from both domains - {domain1.value} indicators {', '.join(concept1.indicators[:2])} with {domain2.value} patterns {', '.join(concept2.indicators[:2])}, 4) Synthesize results to identify systemic vulnerabilities that span multiple domains, 5) Prioritize remediation based on cross-domain impact and exploit chains.",
                            reasoning_type=ReasoningType.SYNTHESIS,
                            security_domain=domain1,  # Primary domain
                            concepts_used=[concept1.name, concept2.name],
                            difficulty_level="expert",
                            source_tool="cross_domain"
                        ))
        
        return examples[:15]  # Limit output
    
    def save_dataset(self, output_path: str = "comprehensive_security_training.jsonl"):
        """Save the complete dataset"""
        output_file = Path(output_path)
        
        # Convert to dictionaries with metadata
        dataset = []
        for example in self.all_examples:
            example_dict = example.to_dict()
            example_dict['dataset_version'] = "comprehensive_security_v1"
            example_dict['generation_timestamp'] = time.time()
            dataset.append(example_dict)
        
        # Save as JSONL for LoRA training
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in dataset:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(dataset)} training examples to {output_file}")
        
        # Generate statistics
        self._print_dataset_statistics()
        
        return output_file
    
    def _print_dataset_statistics(self):
        """Print comprehensive dataset statistics"""
        logger.info("=== Dataset Statistics ===")
        
        # Count by source
        source_counts = {}
        for example in self.all_examples:
            source = example.source_tool
            source_counts[source] = source_counts.get(source, 0) + 1
        
        logger.info("Examples by source:")
        for source, count in sorted(source_counts.items()):
            logger.info(f"  {source}: {count}")
        
        # Count by reasoning type
        reasoning_counts = {}
        for example in self.all_examples:
            reasoning = example.reasoning_type.value
            reasoning_counts[reasoning] = reasoning_counts.get(reasoning, 0) + 1
        
        logger.info("Examples by reasoning type:")
        for reasoning, count in sorted(reasoning_counts.items()):
            logger.info(f"  {reasoning}: {count}")
        
        # Count by domain
        domain_counts = {}
        for example in self.all_examples:
            domain = example.security_domain.value
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        logger.info("Examples by security domain:")
        for domain, count in sorted(domain_counts.items()):
            logger.info(f"  {domain}: {count}")
        
        # Count by difficulty
        difficulty_counts = {}
        for example in self.all_examples:
            difficulty = example.difficulty_level
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        
        logger.info("Examples by difficulty:")
        for difficulty, count in sorted(difficulty_counts.items()):
            logger.info(f"  {difficulty}: {count}")

def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive security training data for LoRA fine-tuning")
    parser.add_argument("--data-path", default="./security_data", help="Base path for data storage")
    parser.add_argument("--output", default="comprehensive_security_training.jsonl", help="Output file path")
    parser.add_argument("--setup", action="store_true", help="Download/clone all data sources")
    parser.add_argument("--sources", nargs='+', 
                        choices=['nuclei', 'bugbounty', 'rfc', 'mdn', 'weird_proxies', 'all'],
                        default=['all'], help="Which sources to process")
    
    args = parser.parse_args()
    
    generator = ComprehensiveSecurityTrainingGenerator(args.data_path)
    
    # Filter extractors based on sources argument
    if 'all' not in args.sources:
        filtered_extractors = {k: v for k, v in generator.extractors.items() if k in args.sources}
        generator.extractors = filtered_extractors
    
    if args.setup:
        generator.setup_data_sources()
    
    # Generate dataset
    dataset = generator.generate_comprehensive_dataset()
    output_file = generator.save_dataset(args.output)
    
    logger.info(f"Comprehensive security training dataset saved to {output_file}")
    logger.info("Dataset is optimized for LoRA fine-tuning with reasoning-based examples")

if __name__ == "__main__":
    main()