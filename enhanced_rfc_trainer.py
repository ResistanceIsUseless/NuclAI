#!/usr/bin/env python3
"""
Enhanced RFC-Based Vulnerability Training Generator v2.0
Incorporates real-world CVEs, server-specific behaviors, and advanced attack patterns
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import requests

@dataclass
class CVEMapping:
    """CVE to RFC violation mapping"""
    cve_id: str
    rfc_violation: str
    description: str
    affected_servers: List[str]
    attack_pattern: str
    detection_method: str
    payload_examples: List[str]
    nuclei_template: str = ""

@dataclass
class ServerBehavior:
    """Server-specific RFC compliance behavior"""
    server_name: str
    version_range: str
    rfc_compliance_level: str
    known_violations: List[str]
    security_implications: List[str]
    detection_signatures: List[str]

class EnhancedRFCTrainingGenerator:
    def __init__(self):
        self.cve_mappings = {}
        self.server_behaviors = {}
        self.advanced_patterns = {}
        self.training_examples = []
        
    def load_real_world_cves(self):
        """Load real-world CVEs mapped to RFC violations"""
        
        cve_database = {
            "CVE-2019-16276": CVEMapping(
                cve_id="CVE-2019-16276",
                rfc_violation="RFC 7230 Section 3.2.6 - Invalid header field syntax",
                description="Go HTTP library accepts spaces after header names",
                affected_servers=["Go net/http", "Applications using Go stdlib"],
                attack_pattern="CL.TE and TE.CL request smuggling",
                detection_method="Check for 'header : value' with space after header name",
                payload_examples=[
                    "Content-Length : 13\\r\\nTransfer-Encoding: chunked",
                    "Transfer-Encoding : chunked\\r\\nContent-Length: 0"
                ]
            ),
            
            "CVE-2022-20713": CVEMapping(
                cve_id="CVE-2022-20713", 
                rfc_violation="RFC 7230 - Improper HTTP request validation",
                description="Cisco ASA VPN web client request smuggling",
                affected_servers=["Cisco ASA", "Cisco FTD"],
                attack_pattern="Browser-powered desync via client-side smuggling",
                detection_method="Monitor for malformed HTTP requests in VPN context",
                payload_examples=[
                    "Malformed Content-Length with browser-specific parsing",
                    "Client-side chunked encoding manipulation"
                ]
            ),
            
            "CVE-2022-42252": CVEMapping(
                cve_id="CVE-2022-42252",
                rfc_violation="RFC 7230 Section 3.3.3 - Content-Length processing",
                description="Apache Tomcat invalid Content-Length handling",
                affected_servers=["Apache Tomcat 8.5.0-8.5.82", "Apache Tomcat 9.0.0-9.0.65"],
                attack_pattern="Content-Length header manipulation",
                detection_method="Validate Content-Length header format strictly",
                payload_examples=[
                    "Content-Length: 13, 13",  # Multiple values
                    "Content-Length: +13",     # Plus prefix
                    "Content-Length: 0x0d"     # Hex format
                ]
            ),
            
            "CVE-2021-33880": CVEMapping(
                cve_id="CVE-2021-33880",
                rfc_violation="RFC 6455 Section 5.1 - Frame masking requirements",
                description="Python websockets timing attack vulnerability",
                affected_servers=["Python websockets library"],
                attack_pattern="WebSocket frame masking timing analysis",
                detection_method="Monitor WebSocket handshake validation timing",
                payload_examples=[
                    "Unmasked client frames to measure timing differences",
                    "Specially crafted masking keys for timing analysis"
                ]
            ),
            
            "CVE-2022-32213": CVEMapping(
                cve_id="CVE-2022-32213",
                rfc_violation="RFC 7230 Section 3.2.6 - CRLF sequence handling",
                description="Node.js llhttp parser bypass via crafted headers",
                affected_servers=["Node.js 14.x, 16.x, 18.x"],
                attack_pattern="Header injection via CRLF bypass",
                detection_method="Strict CRLF validation in header parsing",
                payload_examples=[
                    "Header: value\\r\\n\\r\\nSMUGGLED",
                    "Header: value\\x0d\\x0a\\x0d\\x0aSMUGGLED"
                ]
            )
        }
        
        self.cve_mappings = cve_database
        print(f"✅ Loaded {len(cve_database)} real-world CVE mappings")
    
    def load_server_behaviors(self):
        """Load server-specific RFC compliance behaviors"""
        
        server_database = {
            "apache_httpd": ServerBehavior(
                server_name="Apache HTTP Server",
                version_range="2.4.x",
                rfc_compliance_level="Strict (with HttpProtocolOptions Strict)",
                known_violations=[
                    "Default configuration allows some RFC violations",
                    ".htaccess can override strict parsing",
                    "mod_rewrite can introduce parsing inconsistencies"
                ],
                security_implications=[
                    "Request smuggling possible with default config",
                    "Header injection via .htaccess misconfiguration",
                    "URL parsing differences with mod_rewrite"
                ],
                detection_signatures=[
                    "Server: Apache/2.4.x",
                    "HttpProtocolOptions configuration detection",
                    "mod_rewrite RewriteRule patterns"
                ]
            ),
            
            "nginx": ServerBehavior(
                server_name="Nginx",
                version_range="1.20.x+",
                rfc_compliance_level="Permissive (pre-RFC 7230)",
                known_violations=[
                    "Accepts whitespace between header name and colon",
                    "Tolerates some malformed chunked encoding",
                    "Different Connection header processing"
                ],
                security_implications=[
                    "Potential CL.TE smuggling with backend servers",
                    "Header injection through permissive parsing",
                    "Cache poisoning via header normalization differences"
                ],
                detection_signatures=[
                    "Server: nginx/1.x",
                    "Permissive header parsing behavior",
                    "nginx-specific error pages"
                ]
            ),
            
            "microsoft_iis": ServerBehavior(
                server_name="Microsoft IIS",
                version_range="10.0+",
                rfc_compliance_level="Microsoft-specific extensions",
                known_violations=[
                    "HTTP.sys parsing differences",
                    "Range header processing variations",
                    "WebDAV extension interactions"
                ],
                security_implications=[
                    "DoS via malformed Range headers (MS15-034)",
                    "Authentication bypass via header manipulation",
                    "Path traversal via URL encoding differences"
                ],
                detection_signatures=[
                    "Server: Microsoft-IIS/10.0",
                    "X-Powered-By: ASP.NET",
                    "IIS-specific HTTP status responses"
                ]
            ),
            
            "apache_tomcat": ServerBehavior(
                server_name="Apache Tomcat",
                version_range="9.0.x, 10.x",
                rfc_compliance_level="Java-based strict parsing",
                known_violations=[
                    "Content-Length header format validation",
                    "WebSocket upgrade header processing",
                    "AJP connector parsing differences"
                ],
                security_implications=[
                    "Request smuggling via Content-Length bypass",
                    "WebSocket hijacking through upgrade manipulation",
                    "AJP injection attacks"
                ],
                detection_signatures=[
                    "Server: Apache-Coyote/1.1",
                    "Tomcat-specific error pages",
                    "Java stack trace indicators"
                ]
            )
        }
        
        self.server_behaviors = server_database
        print(f"✅ Loaded {len(server_database)} server behavior profiles")
    
    def load_advanced_attack_patterns(self):
        """Load advanced attack patterns from recent research"""
        
        advanced_patterns = {
            "browser_powered_desync": {
                "name": "Browser-Powered Desync Attack",
                "description": "Client-side request smuggling using browser parsing differences",
                "rfc_violations": ["RFC 7230 - Message boundary ambiguity"],
                "attack_vectors": [
                    "Browser sends malformed request that proxy accepts",
                    "Backend server interprets request differently", 
                    "Smuggled request affects subsequent legitimate requests"
                ],
                "detection_methods": [
                    "Monitor for unusual browser User-Agent patterns",
                    "Detect timing anomalies in request processing",
                    "Check for unexpected request sequences"
                ],
                "payload_examples": [
                    "GET / HTTP/1.1\\r\\nHost: victim.com\\r\\nContent-Length: 35\\r\\n\\r\\nGET /admin HTTP/1.1\\r\\nHost: victim.com\\r\\n\\r\\n",
                    "Browser-specific chunked encoding with malformed extensions"
                ]
            },
            
            "http2_downgrade_smuggling": {
                "name": "HTTP/2 to HTTP/1.1 Downgrade Smuggling", 
                "description": "Exploit HTTP/2 to HTTP/1.1 translation vulnerabilities",
                "rfc_violations": ["RFC 7540 to RFC 7230 translation ambiguities"],
                "attack_vectors": [
                    "HTTP/2 pseudo-headers translation errors",
                    "Header name case sensitivity differences",
                    "Connection-specific header stripping inconsistencies"
                ],
                "detection_methods": [
                    "Monitor HTTP/2 to HTTP/1.1 proxy translation",
                    "Validate pseudo-header to regular header conversion",
                    "Check for header case normalization issues"
                ],
                "payload_examples": [
                    ":method: GET with additional :method: POST",
                    "HTTP/2 headers with connection-specific values"
                ]
            },
            
            "cache_poisoning_via_normalization": {
                "name": "Cache Poisoning via Header Normalization",
                "description": "Exploit header normalization differences between cache and origin",
                "rfc_violations": ["RFC 7234 - Cache key generation ambiguities"],
                "attack_vectors": [
                    "Header case sensitivity differences in cache keys",
                    "URL encoding normalization variations",
                    "Host header manipulation for cache confusion"
                ],
                "detection_methods": [
                    "Monitor cache hit/miss patterns for anomalies",
                    "Validate cache key generation consistency",
                    "Check for unexpected cached content"
                ],
                "payload_examples": [
                    "X-Forwarded-Host: evil.com vs X-FORWARDED-HOST: evil.com",
                    "Host: victim.com vs Host: VICTIM.COM cache confusion"
                ]
            },
            
            "websocket_upgrade_hijacking": {
                "name": "WebSocket Upgrade Hijacking",
                "description": "Hijack WebSocket connections through upgrade header manipulation",
                "rfc_violations": ["RFC 6455 - WebSocket handshake validation"],
                "attack_vectors": [
                    "Malformed Sec-WebSocket-Key calculation",
                    "Origin header bypass techniques",
                    "Connection upgrade header injection"
                ],
                "detection_methods": [
                    "Validate WebSocket handshake cryptographic calculation",
                    "Strict origin validation",
                    "Monitor for unusual WebSocket traffic patterns"
                ],
                "payload_examples": [
                    "Sec-WebSocket-Accept calculation with modified key",
                    "Origin: null bypass attempts"
                ]
            }
        }
        
        self.advanced_patterns = advanced_patterns
        print(f"✅ Loaded {len(advanced_patterns)} advanced attack patterns")
    
    def generate_cve_based_templates(self) -> List[Dict[str, str]]:
        """Generate nuclei templates based on real CVEs"""
        
        examples = []
        
        for cve_id, cve_mapping in self.cve_mappings.items():
            # Generate the nuclei template for this CVE
            nuclei_template = self._generate_cve_nuclei_template(cve_mapping)
            cve_mapping.nuclei_template = nuclei_template
            
            # Create training examples
            base_instruction = f"Create a nuclei template for {cve_id}"
            examples.append({
                "instruction": base_instruction,
                "response": nuclei_template,
                "category": "real_cve",
                "cve_id": cve_id,
                "rfc_violation": cve_mapping.rfc_violation
            })
            
            # Detailed RFC-specific instruction
            rfc_instruction = f"Generate a nuclei template that detects {cve_mapping.rfc_violation} violations as demonstrated in {cve_id}"
            examples.append({
                "instruction": rfc_instruction,
                "response": nuclei_template,
                "category": "rfc_specific_cve",
                "cve_id": cve_id,
                "rfc_section": cve_mapping.rfc_violation
            })
            
            # Attack pattern instruction
            attack_instruction = f"Create a template to detect {cve_mapping.attack_pattern} vulnerabilities"
            examples.append({
                "instruction": attack_instruction,
                "response": nuclei_template,
                "category": "attack_pattern",
                "attack_type": cve_mapping.attack_pattern
            })
        
        return examples
    
    def generate_server_specific_templates(self) -> List[Dict[str, str]]:
        """Generate server-specific vulnerability templates"""
        
        examples = []
        
        for server_key, server_behavior in self.server_behaviors.items():
            for violation in server_behavior.known_violations:
                for implication in server_behavior.security_implications:
                    
                    template = self._generate_server_specific_template(
                        server_behavior, violation, implication
                    )
                    
                    instruction = f"Create a nuclei template to detect {implication} in {server_behavior.server_name}"
                    
                    examples.append({
                        "instruction": instruction,
                        "response": template,
                        "category": "server_specific",
                        "server": server_behavior.server_name,
                        "violation": violation,
                        "implication": implication
                    })
        
        return examples
    
    def generate_advanced_pattern_templates(self) -> List[Dict[str, str]]:
        """Generate templates for advanced attack patterns"""
        
        examples = []
        
        for pattern_key, pattern in self.advanced_patterns.items():
            template = self._generate_advanced_pattern_template(pattern)
            
            # Basic pattern instruction
            instruction = f"Create a nuclei template for {pattern['name']}"
            examples.append({
                "instruction": instruction,
                "response": template,
                "category": "advanced_pattern",
                "pattern_name": pattern["name"]
            })
            
            # Technical detail instruction
            tech_instruction = f"Generate a template that detects {pattern['description']}"
            examples.append({
                "instruction": tech_instruction,
                "response": template,
                "category": "advanced_technical",
                "pattern_type": pattern_key
            })
        
        return examples
    
    def _generate_cve_nuclei_template(self, cve_mapping: CVEMapping) -> str:
        """Generate nuclei template for specific CVE"""
        
        template_id = cve_mapping.cve_id.lower().replace("-", "_")
        
        # Build HTTP requests based on CVE specifics
        if "request smuggling" in cve_mapping.attack_pattern.lower():
            http_section = self._build_smuggling_requests(cve_mapping)
        elif "websocket" in cve_mapping.attack_pattern.lower():
            http_section = self._build_websocket_requests(cve_mapping)
        else:
            http_section = self._build_generic_requests(cve_mapping)
        
        template = f"""id: {template_id}

info:
  name: {cve_mapping.cve_id} - {cve_mapping.description}
  description: |
    {cve_mapping.description}
    RFC Violation: {cve_mapping.rfc_violation}
    Attack Pattern: {cve_mapping.attack_pattern}
  severity: high
  author: enhanced-rfc-generator
  reference:
    - https://cve.mitre.org/cgi-bin/cvename.cgi?name={cve_mapping.cve_id}
    - https://nvd.nist.gov/vuln/detail/{cve_mapping.cve_id}
  classification:
    cve-id: {cve_mapping.cve_id}
  tags:
    - rfc-violation
    - {cve_mapping.cve_id.lower()}
    - {cve_mapping.attack_pattern.lower().replace(' ', '-')}

{http_section}

    matchers:
      - type: word
        words:
          - "smuggled"
          - "vulnerability"
          - "error"
        condition: or
        
      - type: status
        status:
          - 400
          - 500
          - 502
"""
        
        return template
    
    def _build_smuggling_requests(self, cve_mapping: CVEMapping) -> str:
        """Build HTTP requests for request smuggling CVEs"""
        
        requests = []
        for payload in cve_mapping.payload_examples:
            if "Content-Length" in payload and "Transfer-Encoding" in payload:
                requests.append(f"""  - raw:
      - |
        POST / HTTP/1.1
        Host: {{{{Hostname}}}}
        {payload}
        Connection: keep-alive
        
        0
        
        SMUGGLED""")
        
        return "http:\n" + "\n".join(requests)
    
    def _build_websocket_requests(self, cve_mapping: CVEMapping) -> str:
        """Build WebSocket-specific requests"""
        
        return f"""websocket:
  - url: "ws://{{{{Hostname}}}}/ws"
    headers:
      Origin: "http://{{{{Hostname}}}}"
      Sec-WebSocket-Key: "dGhlIHNhbXBsZSBub25jZQ=="
      
    attack:
      - payload: "malformed_frame_data"
        type: "unmasked_frame"
"""
    
    def _build_generic_requests(self, cve_mapping: CVEMapping) -> str:
        """Build generic HTTP requests"""
        
        return f"""http:
  - method: GET
    path:
      - "{{{{BaseURL}}}}"
    
    headers:
      Test-Header: "rfc_violation_test"
"""
    
    def _generate_server_specific_template(self, server: ServerBehavior, violation: str, implication: str) -> str:
        """Generate server-specific vulnerability template"""
        
        template_id = f"{server.server_name.lower().replace(' ', '_')}_{violation.lower().replace(' ', '_')[:20]}"
        
        return f"""id: {template_id}

info:
  name: {server.server_name} - {implication}
  description: |
    Detects {implication} in {server.server_name}.
    Known violation: {violation}
    Compliance level: {server.rfc_compliance_level}
  severity: medium
  author: enhanced-rfc-generator
  tags:
    - {server.server_name.lower().replace(' ', '-')}
    - rfc-violation
    - server-specific

http:
  - method: GET
    path:
      - "{{{{BaseURL}}}}"
    
    headers:
      User-Agent: "RFC-Violation-Test"
    
    matchers:
      - type: word
        words:
{chr(10).join(f'          - "{sig}"' for sig in server.detection_signatures[:3])}
        condition: or
"""
    
    def _generate_advanced_pattern_template(self, pattern: Dict[str, Any]) -> str:
        """Generate template for advanced attack patterns"""
        
        template_id = pattern["name"].lower().replace(" ", "_").replace("-", "_")
        
        return f"""id: {template_id}

info:
  name: {pattern["name"]}
  description: |
    {pattern["description"]}
    RFC Violations: {', '.join(pattern["rfc_violations"])}
  severity: high
  author: advanced-pattern-generator
  tags:
    - advanced-attack
    - rfc-violation
    - {template_id.replace("_", "-")}

http:
  - method: POST
    path:
      - "{{{{BaseURL}}}}"
    
    body: |
      {pattern["payload_examples"][0] if pattern["payload_examples"] else "test_payload"}
    
    matchers:
      - type: word
        words:
          - "desync"
          - "smuggled"
          - "poisoned"
        condition: or
        
      - type: status
        status:
          - 400
          - 502
"""
    
    def generate_comprehensive_training_data(self) -> List[Dict[str, str]]:
        """Generate comprehensive training dataset"""
        
        print("Generating comprehensive RFC training data...")
        
        # Load all data sources
        self.load_real_world_cves()
        self.load_server_behaviors()
        self.load_advanced_attack_patterns()
        
        training_data = []
        
        # Generate CVE-based examples
        cve_examples = self.generate_cve_based_templates()
        training_data.extend(cve_examples)
        print(f"Generated {len(cve_examples)} CVE-based examples")
        
        # Generate server-specific examples
        server_examples = self.generate_server_specific_templates()
        training_data.extend(server_examples)
        print(f"Generated {len(server_examples)} server-specific examples")
        
        # Generate advanced pattern examples
        advanced_examples = self.generate_advanced_pattern_templates()
        training_data.extend(advanced_examples)
        print(f"Generated {len(advanced_examples)} advanced pattern examples")
        
        self.training_examples = training_data
        print(f"✅ Total training examples: {len(training_data)}")
        
        return training_data
    
    def save_enhanced_training_data(self, output_file: str = "enhanced_rfc_training.jsonl"):
        """Save enhanced training dataset"""
        
        with open(output_file, 'w') as f:
            for example in self.training_examples:
                json.dump(example, f, ensure_ascii=False)
                f.write('\\n')
        
        print(f"✅ Saved {len(self.training_examples)} examples to {output_file}")
        
        # Generate statistics
        categories = {}
        for example in self.training_examples:
            cat = example.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        print("\\n=== Training Data Statistics ===")
        for category, count in sorted(categories.items()):
            print(f"{category}: {count} examples")

def main():
    """Generate enhanced RFC-based training data"""
    
    print("=== Enhanced RFC Training Data Generator v2.0 ===\\n")
    
    generator = EnhancedRFCTrainingGenerator()
    
    # Generate comprehensive training data
    training_data = generator.generate_comprehensive_training_data()
    
    # Save the dataset
    generator.save_enhanced_training_data()
    
    print("\\n=== Key Improvements ===")
    print("✅ Real-world CVE mappings with specific payloads")
    print("✅ Server-specific RFC compliance behaviors")
    print("✅ Advanced attack patterns from recent research")
    print("✅ Browser-powered desync attack detection")
    print("✅ HTTP/2 to HTTP/1.1 downgrade vulnerabilities")
    print("✅ Cache poisoning via normalization differences")
    print("✅ WebSocket upgrade hijacking patterns")
    
    print("\\n🚀 Enhanced dataset ready for training!")

if __name__ == "__main__":
    main()
