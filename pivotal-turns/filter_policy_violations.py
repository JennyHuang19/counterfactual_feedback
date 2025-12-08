#!/usr/bin/env python3
"""
Filter OpenAI policy-violating content from router training data.

This script implements a conservative, multi-tier filtering system to remove
policy violations across 7 categories: jailbreaking, harmful content,
cybersecurity attacks, credentials, financial crimes, dark web, and security
circumvention.

Input files:
    - router-data/train_small.jsonl (500 examples)
    - router-data/val.jsonl (444 examples)

Output files:
    - router-data/train_small_filtered.jsonl
    - router-data/val_filtered.jsonl
    - router-data/filtering_report.txt
    - router-data/filtered_examples.jsonl
"""

import json
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from pathlib import Path


@dataclass
class FilterResult:
    """Result of filtering a single example."""
    filtered: bool
    categories: List[str] = field(default_factory=list)
    matched_patterns: List[str] = field(default_factory=list)


@dataclass
class FilterStats:
    """Statistics for filtering results."""
    total_processed: int = 0
    total_filtered: int = 0
    total_kept: int = 0
    by_category: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


class PolicyViolationDetector:
    """Multi-tier detection engine for policy violations."""

    def __init__(self, conservative: bool = True):
        """
        Initialize detector with pattern definitions.

        Args:
            conservative: If True, use lower thresholds for filtering
        """
        self.conservative = conservative
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile all regex patterns for efficiency."""

        # CRITICAL - Jailbreak/Prompt Injection patterns
        self.jailbreak_patterns = [
            (r'(?i)\b(?:DAN|do\s+anything\s+now)\b', 'DAN_persona'),
            (r'(?i)(?:act\s+as|pretend\s+to\s+be|from\s+now\s+on).*\b(?:unrestricted|without.*(?:restrictions?|limitations?|rules|ethics|filters?))', 'unrestricted_persona'),
            (r'(?i)\b(?:developer\s+mode|admin\s+mode|god\s+mode|debug\s+mode)\b', 'dev_mode'),
            (r'(?i)ignore\s+(?:previous|all|your|earlier)\s+(?:instructions?|rules|guidelines?|constraints?)', 'ignore_instructions'),
            (r'(?i)\b(?:jailbreak|jailbroken|bypass.*(?:filter|safety|guardrail))\b', 'explicit_jailbreak'),
            (r'(?i)(?:you\s+are\s+now|from\s+this\s+point).*(?:free|unrestricted|unfiltered|uncensored)', 'unrestricted_mode'),
        ]

        # CRITICAL - Harmful/Illegal Content patterns
        self.harmful_patterns = [
            # Animal cruelty
            (r'(?i)(?:killing|burning|torturing|poisoning|harming).*(?:fish|animal|pet|creature)', 'animal_harm'),
            (r'(?i)(?:wash(?:ing)?\s+up\s+liquid|poison|toxic|bleach).*(?:kill|harm).*(?:fish|animal)', 'animal_poisoning'),

            # Violence
            (r'(?i)(?:how\s+to|guide\s+to|tutorial\s+(?:on|for)).*(?:kill|murder|harm|hurt|attack|assault)', 'violence_tutorial'),

            # Weapons
            (r'(?i)(?:make|build|create|manufacture|construct).*(?:bomb|explosive|weapon|grenade|IED)', 'weapon_creation'),
            (r'(?i)\b(?:pipe\s+bomb|pressure\s+cooker\s+bomb|molotov\s+cocktail)\b', 'specific_weapons'),

            # Drugs
            (r'(?i)(?:make|cook|synthesize|manufacture|produce).*(?:drug|meth|cocaine|heroin|fentanyl|narcotic)', 'drug_manufacturing'),
            (r'(?i)\b(?:methamphetamine|amphetamine)\b.*(?:synthesis|recipe|how\s+to\s+make)', 'drug_synthesis'),

            # Child exploitation
            (r'(?i)\b(?:child|minor|underage).*(?:sexual|abuse|exploitation|pornography|CSAM)\b', 'child_exploitation'),

            # Human trafficking
            (r'(?i)\b(?:human\s+trafficking|sex\s+trafficking|forced\s+labor)\b', 'trafficking'),
        ]

        # CRITICAL - Cybersecurity Attacks patterns
        self.cybersecurity_patterns = [
            (r'(?i)bypass.*\b(?:cloudflare|firewall|security|authentication|2fa|mfa|captcha)\b', 'bypass_security'),
            (r'(?i)\b(?:SQL\s+injection|SQLi|union\s+select|drop\s+table)\b', 'sql_injection'),
            (r'(?i)brute.?forc(?:e|ing).*\b(?:password|credential|login|authentication|hash)\b', 'brute_force'),
            (r'(?i)\b(?:ransomware|malware|virus|trojan|backdoor|rootkit|keylogger)\b', 'malware'),
            (r'(?i)\b(?:crack|hack|exploit).*\b(?:password|encryption|security|system|network|vulnerability)\b', 'hacking'),
            (r'(?i)\b(?:XSS|cross.?site\s+scripting|CSRF|remote\s+code\s+execution|RCE)\b', 'web_exploits'),
            (r'(?i)\b(?:metasploit|burp\s+suite|sqlmap|nmap|wireshark)\b.*(?:exploit|attack|hack)', 'hacking_tools'),
            (r'(?i)(?:DDoS|denial\s+of\s+service).*(?:attack|tool|script)', 'ddos'),
            (r'(?i)\b(?:phishing|spear.?phishing|social\s+engineering)\b.*(?:attack|campaign|email)', 'phishing'),
        ]

        # HIGH - Credentials/Secrets patterns
        self.credentials_patterns = [
            (r'(?i)(?:api[_\s]?key|secret[_\s]?key|access[_\s]?key|private[_\s]?key)', 'api_keys'),
            (r'(?i)password.*(?:sheet|list|database|dump|file)', 'password_sheets'),
            (r'BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY', 'private_key_format'),
            (r'(?i)(?:auth|bearer|jwt|oauth)[_\s]?token', 'auth_tokens'),
            (r'(?i)(?:steal|harvest|extract|dump|crack).*(?:password|credential|token|key)', 'credential_theft'),
            (r'(?i)remove.*(?:password|protection|encryption).*(?:sheet|excel|pdf|file)', 'remove_protection'),
        ]

        # HIGH - Financial Crimes patterns
        self.financial_patterns = [
            (r'(?i)insider\s+trading', 'insider_trading'),
            (r'(?i)(?:balance\s+sheet|financial\s+report|earnings?\s+report).*\b(?:before|early|prior\s+to|ahead\s+of).*\b(?:public|release|announcement|official)', 'early_financial_info'),
            (r'(?i)money\s+launder(?:ing)?', 'money_laundering'),
            (r'(?i)(?:credit\s+card|bank|banking)\s+fraud', 'financial_fraud'),
            (r'(?i)fraudulent.*(?:transaction|transfer|account|charge)', 'fraud'),
            (r'(?i)(?:pump\s+and\s+dump|market\s+manipulation)', 'market_manipulation'),
        ]

        # HIGH - Dark Web patterns
        self.darkweb_patterns = [
            (r'(?i)\b(?:dark\s+web|darknet|deep\s+web)\b', 'darkweb_reference'),
            (r'\.onion\b', 'tor_address'),
            (r'(?i)(?:silk\s+road|alphabay|empire\s+market|dream\s+market)', 'illegal_marketplace'),
            (r'(?i)(?:buy|sell|purchase|acquire).*(?:stolen|illegal|contraband|counterfeit)', 'illegal_trade'),
            (r'(?i)stolen.*(?:credit\s+card|credential|data|information|identity)', 'stolen_data'),
        ]

        # HIGH - Security Circumvention patterns
        self.security_circumvention_patterns = [
            (r'(?i)disable.*\b(?:firewall|antivirus|anti.?virus|encryption|2fa|mfa|security|protection)\b', 'disable_security'),
            (r'(?i)(?:bypass|circumvent|evade|disable|remove).*\b(?:encryption|authentication|authorization|SSL|TLS)\b', 'bypass_crypto'),
            (r'(?i)evad(?:e|ing).*(?:detection|antivirus|anti.?virus|firewall|security)', 'evade_detection'),
            (r'(?i)(?:privilege\s+escalation|lateral\s+movement|persistence)', 'attack_techniques'),
        ]

        # Compile all patterns
        self.all_patterns = {
            'jailbreak': [(re.compile(p), name) for p, name in self.jailbreak_patterns],
            'harmful_content': [(re.compile(p), name) for p, name in self.harmful_patterns],
            'cybersecurity': [(re.compile(p), name) for p, name in self.cybersecurity_patterns],
            'credentials': [(re.compile(p), name) for p, name in self.credentials_patterns],
            'financial_crimes': [(re.compile(p), name) for p, name in self.financial_patterns],
            'darkweb': [(re.compile(p), name) for p, name in self.darkweb_patterns],
            'security_circumvention': [(re.compile(p), name) for p, name in self.security_circumvention_patterns],
        }

    def detect_violations(self, content: str) -> FilterResult:
        """
        Detect policy violations in content.

        Args:
            content: The text to analyze (full conversation)

        Returns:
            FilterResult with detection results
        """
        categories_triggered = set()
        matched_patterns = []

        # Check all pattern categories
        for category, patterns in self.all_patterns.items():
            for pattern, pattern_name in patterns:
                if pattern.search(content):
                    categories_triggered.add(category)
                    matched_patterns.append(f"{category}::{pattern_name}")
                    break  # One match per category is enough

        return FilterResult(
            filtered=len(categories_triggered) > 0,
            categories=list(categories_triggered),
            matched_patterns=matched_patterns
        )


def extract_content(example: Dict) -> str:
    """
    Extract all text content from a JSONL example.

    Args:
        example: Dictionary with 'messages' field

    Returns:
        Concatenated content from all messages
    """
    parts = []

    for message in example.get('messages', []):
        role = message.get('role', '')
        content = message.get('content', '')
        parts.append(f"{role}: {content}")

    return "\n\n".join(parts)


def process_jsonl_file(
    input_path: Path,
    output_path: Path,
    detector: PolicyViolationDetector
) -> Tuple[FilterStats, List[Dict]]:
    """
    Process a JSONL file and filter policy violations.

    Args:
        input_path: Input JSONL file
        output_path: Output JSONL file (filtered)
        detector: PolicyViolationDetector instance

    Returns:
        Tuple of (FilterStats, list of filtered examples)
    """
    stats = FilterStats()
    filtered_examples = []

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in, 1):
            stats.total_processed += 1

            try:
                example = json.loads(line)

                # Extract all content for analysis
                full_content = extract_content(example)

                # Run detection
                result = detector.detect_violations(full_content)

                if result.filtered:
                    # Filter this example
                    stats.total_filtered += 1
                    for category in result.categories:
                        stats.by_category[category] += 1

                    # Store for reporting
                    filtered_examples.append({
                        'line_number': line_num,
                        'example': example,
                        'categories': result.categories,
                        'patterns': result.matched_patterns
                    })
                else:
                    # Keep this example
                    stats.total_kept += 1
                    f_out.write(line)

            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON at line {line_num}: {e}")
                # In conservative mode, filter invalid JSON
                if detector.conservative:
                    stats.total_filtered += 1
                    stats.by_category['invalid_json'] += 1

    return stats, filtered_examples


def generate_report(
    train_stats: FilterStats,
    # val_stats: FilterStats,
    train_filtered: List[Dict],
    # val_filtered: List[Dict],
    output_path: Path
):
    """
    Generate comprehensive filtering report.

    Args:
        train_stats: Statistics for training set
        val_stats: Statistics for validation set
        train_filtered: List of filtered training examples
        val_filtered: List of filtered validation examples
        output_path: Path to save report
    """
    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("POLICY VIOLATION FILTERING REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Overall statistics
    report_lines.append("OVERALL STATISTICS")
    report_lines.append("-" * 80)
    report_lines.append(f"Training Set:")
    report_lines.append(f"  Total processed: {train_stats.total_processed}")
    report_lines.append(f"  Filtered out: {train_stats.total_filtered} ({train_stats.total_filtered/train_stats.total_processed*100:.1f}%)")
    report_lines.append(f"  Kept: {train_stats.total_kept} ({train_stats.total_kept/train_stats.total_processed*100:.1f}%)")
    report_lines.append("")

    # report_lines.append(f"Validation Set:")
    # report_lines.append(f"  Total processed: {val_stats.total_processed}")
    # report_lines.append(f"  Filtered out: {val_stats.total_filtered} ({val_stats.total_filtered/val_stats.total_processed*100:.1f}%)")
    # report_lines.append(f"  Kept: {val_stats.total_kept} ({val_stats.total_kept/val_stats.total_processed*100:.1f}%)")
    # report_lines.append("")

    # Category breakdown
    report_lines.append("FILTERING BY CATEGORY")
    report_lines.append("-" * 80)

    all_categories = set(train_stats.by_category.keys()) | set(train_stats.by_category.keys())

    report_lines.append(f"{'Category':<30} {'Train':<10} {'Val':<10}")
    report_lines.append("-" * 50)

    for category in sorted(all_categories):
        train_count = train_stats.by_category.get(category, 0)
        # val_count = val_stats.by_category.get(category, 0)
        report_lines.append(f"{category:<30} {train_count:<10}") # {val_count:<10}

    report_lines.append("")

    # Sample filtered examples
    report_lines.append("SAMPLE FILTERED EXAMPLES (First 5 from training set)")
    report_lines.append("-" * 80)

    for i, item in enumerate(train_filtered[:5], 1):
        report_lines.append(f"\nExample {i} (Line {item['line_number']}):")
        report_lines.append(f"Categories: {', '.join(item['categories'])}")
        report_lines.append(f"Patterns matched: {', '.join(item['patterns'][:3])}")

        # Show snippet of content
        content = extract_content(item['example'])
        snippet = content[:200] + "..." if len(content) > 200 else content
        report_lines.append(f"Content preview: {snippet}")

    report_lines.append("")
    report_lines.append("=" * 80)

    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    print(f"\nReport saved to: {output_path}")


def save_filtered_examples(
    train_filtered: List[Dict],
    # val_filtered: List[Dict],
    output_path: Path
):
    """
    Save filtered examples to JSONL for review.

    Args:
        train_filtered: Filtered training examples
        val_filtered: Filtered validation examples
        output_path: Path to save filtered examples
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in train_filtered:
            record = {
                'source': 'train',
                'line_number': item['line_number'],
                'categories': item['categories'],
                'patterns': item['patterns'],
                'example': item['example']
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

        # for item in val_filtered:
        #     record = {
        #         'source': 'val',
        #         'line_number': item['line_number'],
        #         'categories': item['categories'],
        #         'patterns': item['patterns'],
        #         'example': item['example']
        #     }
        #     f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Filtered examples saved to: {output_path}")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Filter policy violations from router training data'
    )
    parser.add_argument(
        '--conservative',
        action='store_true',
        default=True,
        help='Use conservative filtering (default: True)'
    )
    parser.add_argument(
        '--save-filtered',
        action='store_true',
        default=True,
        help='Save filtered examples for review (default: True)'
    )
    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'router-data'

    train_input = '/dccstor/gma2/jhjenny9/counterfactual_feedback/router-data/train.jsonl'
    train_input = Path(train_input)
    train_output = '/dccstor/gma2/jhjenny9/counterfactual_feedback/router-data/train_filtered.jsonl'
    train_output = Path(train_output)

    # val_input = data_dir / 'val.jsonl'
    # val_output = data_dir / 'val_filtered.jsonl'

    report_path = data_dir / 'filtering_report.txt'
    filtered_examples_path = data_dir / 'filtered_examples.jsonl'

    print("=" * 80)
    print("POLICY VIOLATION FILTERING")
    print("=" * 80)
    print(f"\nMode: {'CONSERVATIVE' if args.conservative else 'MODERATE'}")
    print(f"Input files:")
    print(f"  Train: {train_input} ({train_input.stat().st_size / 1024:.1f} KB)")
    # print(f"  Val: {val_input} ({val_input.stat().st_size / 1024:.1f} KB)")
    print("")

    # Initialize detector
    print("Initializing policy violation detector...")
    detector = PolicyViolationDetector(conservative=args.conservative)
    total_patterns = sum(len(patterns) for patterns in detector.all_patterns.values())
    print(f"  Loaded {total_patterns} detection patterns across {len(detector.all_patterns)} categories")
    print("")

    # Process training set
    print("Processing training set...")
    train_stats, train_filtered = process_jsonl_file(
        train_input,
        train_output,
        detector
    )
    print(f"  Filtered: {train_stats.total_filtered}/{train_stats.total_processed} "
          f"({train_stats.total_filtered/train_stats.total_processed*100:.1f}%)")
    print("")

    # Process validation set
    # print("Processing validation set...")
    # val_stats, val_filtered = process_jsonl_file(
    #     val_input,
    #     val_output,
    #     detector
    # )
    # print(f"  Filtered: {val_stats.total_filtered}/{val_stats.total_processed} "
    #       f"({val_stats.total_filtered/val_stats.total_processed*100:.1f}%)")
    # print("")

    # Generate reports
    print("Generating reports...")
    generate_report(
        train_stats,
        # val_stats,
        train_filtered,
        # val_filtered,
        report_path
    )

    if args.save_filtered:
        save_filtered_examples(
            train_filtered,
            # val_filtered,
            filtered_examples_path
        )

    print("")
    print("=" * 80)
    print("FILTERING COMPLETE")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  {train_output}")
    # print(f"  {val_output}")
    print(f"  {report_path}")
    if args.save_filtered:
        print(f"  {filtered_examples_path}")
    print("")


if __name__ == '__main__':
    main()
