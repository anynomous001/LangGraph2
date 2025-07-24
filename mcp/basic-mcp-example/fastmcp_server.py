#!/usr/bin/env python3
"""
Advanced MCP Server using FastMCP
Provides web search, PDF reading, calculator, and file operations tools
"""

import asyncio
import aiohttp
import PyPDF2
import json
import math
import os
import io
from typing import Dict, Any, List, Optional
from pathlib import Path
import re
from datetime import datetime
import hashlib

from fastmcp import FastMCP

# Create FastMCP server instance
mcp = FastMCP("Advanced MCP Server")

@mcp.tool()
async def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for information using DuckDuckGo Instant Answer API
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        Formatted search results as string
    """
    try:
        # Using DuckDuckGo Instant Answer API (no API key required)
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    results = []
                    
                    # Add instant answer if available
                    if data.get("Answer"):
                        results.append(f"Direct Answer: {data['Answer']}")
                    
                    if data.get("Definition"):
                        results.append(f"Definition: {data['Definition']}")
                    
                    # Add abstract if available
                    if data.get("Abstract"):
                        results.append(f"Summary: {data['Abstract']}")
                    
                    # Add related topics
                    related_topics = data.get("RelatedTopics", [])[:max_results]
                    for i, topic in enumerate(related_topics, 1):
                        if isinstance(topic, dict) and topic.get("Text"):
                            results.append(f"Result {i}: {topic['Text']}")
                            if topic.get("FirstURL"):
                                results.append(f"Source: {topic['FirstURL']}")
                    
                    if results:
                        return "\n\n".join(results)
                    else:
                        return f"No detailed results found for '{query}'. Try a more specific search term."
                else:
                    return f"Search failed with status code: {response.status}"
                    
    except Exception as e:
        return f"Search error: {str(e)}"

@mcp.tool()
async def read_pdf(file_path: str, page_range: Optional[str] = None) -> str:
    """
    Read and extract text from a PDF file
    
    Args:
        file_path: Path to the PDF file
        page_range: Optional page range (e.g., "1-3" or "5" or "1,3,5")
    
    Returns:
        Extracted text from the PDF
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' not found"
        
        if not file_path.lower().endswith('.pdf'):
            return f"Error: '{file_path}' is not a PDF file"
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            # Parse page range
            pages_to_read = []
            if page_range:
                try:
                    if '-' in page_range:
                        start, end = map(int, page_range.split('-'))
                        pages_to_read = list(range(start-1, min(end, total_pages)))
                    elif ',' in page_range:
                        pages_to_read = [int(p.strip())-1 for p in page_range.split(',')]
                        pages_to_read = [p for p in pages_to_read if 0 <= p < total_pages]
                    else:
                        page_num = int(page_range) - 1
                        if 0 <= page_num < total_pages:
                            pages_to_read = [page_num]
                except ValueError:
                    return f"Error: Invalid page range format '{page_range}'"
            else:
                pages_to_read = list(range(total_pages))
            
            if not pages_to_read:
                return "Error: No valid pages to read"
            
            # Extract text
            extracted_text = []
            for page_num in pages_to_read:
                try:
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text.strip():
                        extracted_text.append(f"--- Page {page_num + 1} ---\n{text}")
                except Exception as e:
                    extracted_text.append(f"--- Page {page_num + 1} ---\nError reading page: {str(e)}")
            
            result = "\n\n".join(extracted_text)
            return result if result.strip() else "No readable text found in the specified pages"
            
    except Exception as e:
        return f"PDF reading error: {str(e)}"

@mcp.tool()
async def calculator(expression: str) -> str:
    """
    Advanced calculator that can handle mathematical expressions, functions, and constants
    
    Args:
        expression: Mathematical expression to evaluate
    
    Returns:
        Result of the calculation
    """
    try:
        # Clean the expression
        expression = expression.strip()
        
        # Replace common math functions and constants
        replacements = {
            'π': 'math.pi',
            'pi': 'math.pi',
            'e': 'math.e',
            'sin': 'math.sin',
            'cos': 'math.cos',
            'tan': 'math.tan',
            'log': 'math.log10',
            'ln': 'math.log',
            'sqrt': 'math.sqrt',
            'abs': 'abs',
            'pow': 'pow',
            'exp': 'math.exp',
            'factorial': 'math.factorial',
            'degrees': 'math.degrees',
            'radians': 'math.radians',
        }
        
        for old, new in replacements.items():
            expression = re.sub(r'\b' + old + r'\b', new, expression)
        
        # Create safe evaluation environment
        safe_dict = {
            "__builtins__": {},
            "math": math,
            "abs": abs,
            "pow": pow,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
        }
        
        # Evaluate the expression
        result = eval(expression, safe_dict)
        
        # Format the result
        if isinstance(result, float):
            if result.is_integer():
                result = int(result)
            else:
                result = round(result, 10)  # Limit decimal places
        
        return f"Expression: {expression}\nResult: {result}"
        
    except ZeroDivisionError:
        return "Error: Division by zero"
    except ValueError as e:
        return f"Error: Invalid mathematical operation - {str(e)}"
    except SyntaxError:
        return f"Error: Invalid expression syntax - '{expression}'"
    except Exception as e:
        return f"Calculation error: {str(e)}"

@mcp.tool()
async def file_operations(operation: str, file_path: str, content: Optional[str] = None) -> str:
    """
    Perform file operations like read, write, create, delete, and list
    
    Args:
        operation: Operation to perform (read, write, create, delete, list, info)
        file_path: Path to the file or directory
        content: Content to write (required for write/create operations)
    
    Returns:
        Result of the file operation
    """
    try:
        path = Path(file_path)
        
        if operation == "read":
            if not path.exists():
                return f"Error: File '{file_path}' does not exist"
            if path.is_dir():
                return f"Error: '{file_path}' is a directory, not a file"
            
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return f"File: {file_path}\nSize: {len(content)} characters\n\nContent:\n{content}"
            except UnicodeDecodeError:
                return f"Error: Cannot read '{file_path}' - file appears to be binary"
        
        elif operation == "write":
            if content is None:
                return "Error: Content is required for write operation"
            
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote {len(content)} characters to '{file_path}'"
        
        elif operation == "create":
            if path.exists():
                return f"Error: '{file_path}' already exists"
            
            if content is None:
                content = ""
            
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully created file '{file_path}'"
        
        elif operation == "delete":
            if not path.exists():
                return f"Error: '{file_path}' does not exist"
            
            if path.is_file():
                path.unlink()
                return f"Successfully deleted file '{file_path}'"
            elif path.is_dir():
                if any(path.iterdir()):
                    return f"Error: Directory '{file_path}' is not empty"
                path.rmdir()
                return f"Successfully deleted empty directory '{file_path}'"
        
        elif operation == "list":
            if not path.exists():
                return f"Error: Path '{file_path}' does not exist"
            
            if path.is_file():
                return f"'{file_path}' is a file, not a directory"
            
            items = []
            for item in sorted(path.iterdir()):
                item_type = "DIR" if item.is_dir() else "FILE"
                size = ""
                if item.is_file():
                    try:
                        size = f" ({item.stat().st_size} bytes)"
                    except:
                        size = ""
                items.append(f"{item_type}: {item.name}{size}")
            
            if items:
                return f"Contents of '{file_path}':\n" + "\n".join(items)
            else:
                return f"Directory '{file_path}' is empty"
        
        elif operation == "info":
            if not path.exists():
                return f"Error: Path '{file_path}' does not exist"
            
            stat = path.stat()
            info = [
                f"Path: {file_path}",
                f"Type: {'Directory' if path.is_dir() else 'File'}",
                f"Size: {stat.st_size} bytes",
                f"Modified: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}",
                f"Created: {datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')}",
            ]
            return "\n".join(info)
        
        else:
            return f"Error: Unknown operation '{operation}'. Supported: read, write, create, delete, list, info"
    
    except PermissionError:
        return f"Error: Permission denied for '{file_path}'"
    except Exception as e:
        return f"File operation error: {str(e)}"

@mcp.tool()
async def text_analyzer(text: str, analysis_type: str = "all") -> str:
    """
    Analyze text for various metrics and properties
    
    Args:
        text: Text content to analyze
        analysis_type: Type of analysis (all, basic, readability, words, sentences)
    
    Returns:
        Text analysis results
    """
    try:
        results = []
        
        if analysis_type in ["all", "basic"]:
            # Basic statistics
            char_count = len(text)
            char_count_no_spaces = len(text.replace(' ', ''))
            word_count = len(text.split())
            sentence_count = len([s for s in re.split(r'[.!?]+', text) if s.strip()])
            paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
            
            results.append("=== BASIC STATISTICS ===")
            results.append(f"Characters: {char_count:,}")
            results.append(f"Characters (no spaces): {char_count_no_spaces:,}")
            results.append(f"Words: {word_count:,}")
            results.append(f"Sentences: {sentence_count:,}")
            results.append(f"Paragraphs: {paragraph_count:,}")
            
            if word_count > 0:
                avg_word_length = char_count_no_spaces / word_count
                results.append(f"Average word length: {avg_word_length:.1f} characters")
            
            if sentence_count > 0:
                avg_words_per_sentence = word_count / sentence_count
                results.append(f"Average words per sentence: {avg_words_per_sentence:.1f}")
        
        if analysis_type in ["all", "words"]:
            # Word frequency analysis
            words = re.findall(r'\b\w+\b', text.lower())
            if words:
                word_freq = {}
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1
                
                most_common = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                
                results.append("\n=== MOST FREQUENT WORDS ===")
                for word, count in most_common:
                    results.append(f"{word}: {count}")
        
        if analysis_type in ["all", "readability"]:
            # Simple readability metrics
            words = text.split()
            sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
            
            if len(words) > 0 and len(sentences) > 0:
                avg_sentence_length = len(words) / len(sentences)
                
                # Count syllables (simple approximation)
                def count_syllables(word):
                    word = word.lower().strip()
                    vowels = 'aeiouy'
                    syllable_count = 0
                    prev_was_vowel = False
                    
                    for char in word:
                        if char in vowels:
                            if not prev_was_vowel:
                                syllable_count += 1
                            prev_was_vowel = True
                        else:
                            prev_was_vowel = False
                    
                    if word.endswith('e'):
                        syllable_count -= 1
                    
                    return max(1, syllable_count)
                
                total_syllables = sum(count_syllables(word) for word in words)
                avg_syllables_per_word = total_syllables / len(words)
                
                # Flesch Reading Ease (simplified)
                flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
                
                results.append("\n=== READABILITY METRICS ===")
                results.append(f"Average sentence length: {avg_sentence_length:.1f} words")
                results.append(f"Average syllables per word: {avg_syllables_per_word:.1f}")
                results.append(f"Flesch Reading Ease: {flesch_score:.1f}")
                
                if flesch_score >= 90:
                    level = "Very Easy"
                elif flesch_score >= 80:
                    level = "Easy"
                elif flesch_score >= 70:
                    level = "Fairly Easy"
                elif flesch_score >= 60:
                    level = "Standard"
                elif flesch_score >= 50:
                    level = "Fairly Difficult"
                elif flesch_score >= 30:
                    level = "Difficult"
                else:
                    level = "Very Difficult"
                
                results.append(f"Reading Level: {level}")
        
        return "\n".join(results)
        
    except Exception as e:
        return f"Text analysis error: {str(e)}"

@mcp.tool()
async def hash_generator(text: str, hash_type: str = "md5") -> str:
    """
    Generate various types of hashes for the given text
    
    Args:
        text: Text to hash
        hash_type: Type of hash (md5, sha1, sha256, sha512, all)
    
    Returns:
        Generated hash(es)
    """
    try:
        results = []
        
        if hash_type in ["all", "md5"]:
            md5_hash = hashlib.md5(text.encode()).hexdigest()
            results.append(f"MD5: {md5_hash}")
        
        if hash_type in ["all", "sha1"]:
            sha1_hash = hashlib.sha1(text.encode()).hexdigest()
            results.append(f"SHA1: {sha1_hash}")
        
        if hash_type in ["all", "sha256"]:
            sha256_hash = hashlib.sha256(text.encode()).hexdigest()
            results.append(f"SHA256: {sha256_hash}")
        
        if hash_type in ["all", "sha512"]:
            sha512_hash = hashlib.sha512(text.encode()).hexdigest()
            results.append(f"SHA512: {sha512_hash}")
        
        if not results:
            return f"Error: Unsupported hash type '{hash_type}'. Supported: md5, sha1, sha256, sha512, all"
        
        return f"Hash results for text (length: {len(text)} chars):\n" + "\n".join(results)
        
    except Exception as e:
        return f"Hash generation error: {str(e)}"

if __name__ == "__main__":
    # Run the FastMCP server
    mcp.run()