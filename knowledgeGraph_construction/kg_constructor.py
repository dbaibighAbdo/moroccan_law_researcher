import os
from dotenv import load_dotenv
load_dotenv()

import asyncio

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter

# tag::import_loader[]
from neo4j_graphrag.experimental.components.pdf_loader import PdfLoader, PdfDocument

import re
from fsspec import AbstractFileSystem
from typing import Dict, Optional, Union
from pathlib import Path
# end::import_loader[]

neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
neo4j_driver.verify_connectivity()

llm = OpenAILLM(
    model_name="gpt-4o-mini",
    model_params={
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
)

embedder = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

text_splitter = FixedSizeSplitter(chunk_size=800, chunk_overlap=100)

# tag::loader[]
class CustomPDFLoader(PdfLoader):
    async def run(
        self,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, str]] = None,
        fs: Optional[Union[AbstractFileSystem, str]] = None,
    ) -> PdfDocument:
        pdf_document = await super().run(filepath, metadata, fs)

        # Process the PDF document
        # remove asciidoc attribute lines like :id:
        pdf_document.text = re.sub(r':*:.*\n?', '', pdf_document.text, flags=re.MULTILINE)

        return pdf_document

data_loader = CustomPDFLoader()
# end::loader[]



# end of knowledge graph schema definition
PROMPT = """ 
  You are an information extraction system.
The document is written in French.

Return ONLY a valid JSON object.
Do not add new lines before the first character.
Do not wrap the output in markdown.
Do not explain anything.

The JSON MUST start with { and end with }.

FORMAT (MANDATORY):

{"nodes":[{"label":"Article","properties":{"number":"STRING","text":"STRING"}}],"relationships":[{"source":{"label":"Article","key":"number","value":"STRING"},"type":"GRANTS_RIGHT","target":{"label":"Right","properties":{"name":"STRING"}}}]}

RULES:
- Use only the allowed labels and relationship types.
- Keep Arabic text exactly as in the document.
- Always include both keys: "nodes" and "relationships".
- If nothing is found, return exactly:

{"nodes":[],"relationships":[]}

 """
# tag::kg_builder[]
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver, 
    neo4j_database=os.getenv("NEO4J_DATABASE"), 
    embedder=embedder, 
    from_pdf=True,
    text_splitter=text_splitter,
    pdf_loader=data_loader
)
# end::kg_builder[]

# tag::run_loader[]
pdf_file = "data/code_de_travail_marocain.pdf"
doc = asyncio.run(data_loader.run(pdf_file))
print(doc.text)
# end::run_loader[]

print(f"Processing {pdf_file}")
result = asyncio.run(kg_builder.run_async(file_path=pdf_file))
print(result.result)







