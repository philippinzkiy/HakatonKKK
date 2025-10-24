import streamlit as st
import pandas as pd
from docx import Document
import io
import re
import json
from datetime import datetime
import requests
import pdfplumber

class OllamaClient:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.model = "mistral:7b-instruct"
    
    def is_available(self):
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_response(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 1000}
        }
        
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            return None
        except:
            return None

class DataLoader:
    def __init__(self):
        self.products_df = None
    
    def load_products_from_files(self, uploaded_files):
        try:
            all_dfs = []
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name.lower()
                
                if file_name.endswith('.csv'):
                    df = self._load_csv(uploaded_file)
                elif file_name.endswith('.pdf'):
                    df = self._load_pdf(uploaded_file)
                elif file_name.endswith('.json'):
                    # JSON файлы с товарами пока не поддерживаем для каталога
                    st.warning(f"JSON файлы с запросами обрабатываются отдельно. Файл {uploaded_file.name} пропущен.")
                    continue
                else:
                    st.error(f"Неподдерживаемый формат файла: {uploaded_file.name}")
                    continue
                
                if df is not None:
                    all_dfs.append(df)
            
            if all_dfs:
                self.products_df = pd.concat(all_dfs, ignore_index=True)
                self.products_df = self.products_df.drop_duplicates(subset=['Товар'])
                return True
            return False
                
        except Exception as e:
            st.error(f"Ошибка загрузки файлов: {str(e)}")
            return False
    
    def _load_csv(self, uploaded_file):
        df = pd.read_csv(uploaded_file)
        if 'Товар' not in df.columns:
            for col in df.columns:
                if any(keyword in str(col).lower() for keyword in ['товар', 'product', 'наименование']):
                    df = df.rename(columns={col: 'Товар'})
                    break
            else:
                df.columns = ['Товар'] + list(df.columns[1:])
        return df
    
    def _load_pdf(self, uploaded_file):
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                text_content = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_content.append(text)
            
            full_text = "\n".join(text_content)
            
            # Улучшенное извлечение товаров из PDF
            product_lines = []
            
            # Разделяем текст на строки и обрабатываем каждую строку
            lines = full_text.split('\n')
            current_line = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Если строка содержит "руб", это может быть товар
                if 'руб' in line.lower():
                    # Если у нас есть накопленная предыдущая строка, добавляем ее к текущей
                    if current_line:
                        line = current_line + " " + line
                        current_line = ""
                    
                    # Очищаем строку от лишних пробелов
                    clean_line = re.sub(r'\s+', ' ', line.strip())
                    product_lines.append(clean_line)
                else:
                    # Если строка не содержит "руб", но может быть частью описания товара
                    # Накопляем ее для следующей итерации
                    current_line = line
            
            # Добавляем последнюю накопленную строку, если она есть
            if current_line and 'руб' in current_line.lower():
                clean_line = re.sub(r'\s+', ' ', current_line.strip())
                product_lines.append(clean_line)
            
            if product_lines:
                df = pd.DataFrame(product_lines, columns=['Товар'])
                return df
            else:
                st.warning(f"В PDF файле {uploaded_file.name} не найдено товаров с ценами")
                return None
                
        except Exception as e:
            st.error(f"Ошибка чтения PDF файла {uploaded_file.name}: {str(e)}")
            return None
    
    def parse_product_info(self, product_string):
        try:
            product_str = str(product_string)
            price_match = re.search(r'(\d+(?:\s?\d+)*)\s*руб', product_str)
            price = price_match.group(1).replace(' ', '') if price_match else "0"
            return {'name': product_str, 'price': int(price)}
        except:
            return {'name': str(product_string), 'price': 0}

class AISearch:
    def __init__(self, ollama_client, data_loader):
        self.ollama = ollama_client
        self.data_loader = data_loader
    
    def find_products(self, query):
        if not self.ollama.is_available():
            st.error("Ollama не доступен. Запустите: ollama serve")
            return []
        
        if self.data_loader.products_df is None:
            st.error("Нет загруженных данных о товарах")
            return []
        
        products_list = [str(p) for p in self.data_loader.products_df['Товар'].head(100)]
        products_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(products_list)])
        
        prompt = f"""
        Запрос клиента: "{query}"
        
        Доступные товары:
        {products_text}
        
        Выбери от 3 до 8 наиболее подходящих товаров для этого запроса.
        Сначала проанализируй запрос и определи какие типы товаров нужны.
        Затем найди конкретные товары которые соответствуют требованиям.
        
        Верни ТОЛЬКО номера выбранных товаров через запятую в формате: 1, 3, 5
        Не добавляй никакого дополнительного текста.
        """
        
        response = self.ollama.generate_response(prompt)
        
        if response:
            numbers = []
            number_patterns = [
                r'\b\d+\b',
                r'№\s*(\d+)',
                r'номер\s*(\d+)',
            ]
            
            for pattern in number_patterns:
                found = re.findall(pattern, response)
                numbers.extend(found)
            
            selected_indices = []
            for num in set(numbers):
                if num.isdigit():
                    idx = int(num) - 1
                    if 0 <= idx < len(products_list):
                        selected_indices.append(idx)
            
            if selected_indices:
                matches = []
                for idx in selected_indices[:8]:
                    product_str = products_list[idx]
                    product_info = self.data_loader.parse_product_info(product_str)
                    matches.append(product_info)
                return matches
        
        return self._fallback_search(query, products_list)
    
    def _fallback_search(self, query, products_list):
        query_words = query.lower().split()
        matches = []
        
        for i, product_str in enumerate(products_list):
            product_lower = product_str.lower()
            score = 0
            
            for word in query_words:
                if len(word) > 2 and word in product_lower:
                    score += 1
                    if word in ['короб', 'крышка', 'гайка', 'винт', 'лоток', 'хомут']:
                        score += 2
            
            if score > 0:
                product_info = self.data_loader.parse_product_info(product_str)
                matches.append((score, product_info))
        
        matches.sort(key=lambda x: x[0], reverse=True)
        return [product for score, product in matches[:5]]

class DocumentGenerator:
    def generate_tcp_document(self, order_data, products):
        doc = Document()
        
        order_paragraph = doc.add_paragraph()
        order_paragraph.add_run("Заказ № ").bold = True
        order_paragraph.add_run(f"{order_data['order_id']}")
        
        doc.add_paragraph("Перечень комплектующих:")
        
        total_cost = 0
        for product in products:
            p = doc.add_paragraph()
            p.add_run(f"{product['name']}")
            total_cost += product['price']
        
        price_paragraph = doc.add_paragraph()
        price_paragraph.add_run("Итоговая цена: ").bold = True
        price_paragraph.add_run(f"{total_cost} руб.")
        
        return doc, total_cost

class JSONProcessor:
    def __init__(self, ai_search, doc_generator):
        self.ai_search = ai_search
        self.doc_generator = doc_generator
    
    def process_json_file(self, json_file):
        try:
            # Читаем и парсим JSON
            json_data = json.load(json_file)
            
            # Обрабатываем разные форматы JSON
            if isinstance(json_data, list):
                # Если JSON представляет собой массив запросов
                return self._process_multiple_queries(json_data)
            elif isinstance(json_data, dict):
                # Если JSON представляет собой один запрос
                if 'query' in json_data:
                    return self._process_single_query(json_data)
                elif 'queries' in json_data:
                    # Если есть поле queries с массивом запросов
                    return self._process_multiple_queries(json_data['queries'])
                else:
                    st.error("Некорректная структура JSON: отсутствует поле 'query' или 'queries'")
                    return []
            else:
                st.error("Некорректный формат JSON файла")
                return []
                
        except json.JSONDecodeError as e:
            st.error(f"Ошибка парсинга JSON: {str(e)}")
            return []
        except Exception as e:
            st.error(f"Ошибка обработки JSON файла: {str(e)}")
            return []
    
    def _process_single_query(self, query_data):
        """Обработка одного запроса из JSON"""
        query_text = query_data.get('query', '')
        query_id = query_data.get('id', 'unknown')
        
        if not query_text:
            st.error("В JSON отсутствует текст запроса (поле 'query')")
            return []
        
        with st.spinner(f"Обработка запроса {query_id}: {query_text}"):
            matches = self.ai_search.find_products(query_text)
            
            if not matches:
                st.warning(f"Для запроса '{query_text}' не найдено товаров")
                return []
            
            # Создаем документ ТКП
            order_data = {'order_id': f"JSON-{query_id}"}
            doc, total_cost = self.doc_generator.generate_tcp_document(order_data, matches)
            
            # Сохраняем документ в BytesIO
            doc_buffer = io.BytesIO()
            doc.save(doc_buffer)
            doc_buffer.seek(0)
            
            return [{
                'query_id': query_id,
                'query_text': query_text,
                'matches': matches,
                'total_cost': total_cost,
                'document': doc_buffer,
                'filename': f"tcp_json_{query_id}.docx"
            }]
    
    def _process_multiple_queries(self, queries):
        """Обработка массива запросов из JSON"""
        results = []
        
        for query_data in queries:
            if isinstance(query_data, dict) and 'query' in query_data:
                result = self._process_single_query(query_data)
                results.extend(result)
            else:
                st.warning(f"Пропущен некорректный элемент в массиве запросов: {query_data}")
        
        return results

def main():
    st.set_page_config(page_title="Auto-TKP", layout="centered")
    st.title("Auto-TKP Генератор")
    
    ollama_client = OllamaClient()
    data_loader = DataLoader()
    ai_search = AISearch(ollama_client, data_loader)
    doc_generator = DocumentGenerator()
    json_processor = JSONProcessor(ai_search, doc_generator)
    
    # Загрузка каталога товаров
    st.header("1. Загрузка каталога товаров")
    uploaded_files = st.file_uploader(
        "Выберите CSV или PDF файлы с товарами",
        type=['csv', 'pdf'],
        accept_multiple_files=True,
        key="catalog_uploader"
    )
    
    if uploaded_files:
        if data_loader.load_products_from_files(uploaded_files):
            st.success(f"Загружено товаров: {len(data_loader.products_df)}")
    
    # Разделяем интерфейс на две вкладки
    tab1, tab2 = st.tabs(["📝 Ручной ввод запроса", "📄 Обработка JSON запросов"])
    
    with tab1:
        st.header("2. Ручной ввод запроса")
        customer_query = st.text_area(
            "Описание необходимых товаров:",
            placeholder="Мне нужен пластиковый трубопровод длиной 10 м и диаметром не менее 7 см",
            height=80,
            key="manual_query"
        )
        
        order_number = st.text_input("Номер заказа", value=f"TKP-{datetime.now().strftime('%Y%m%d')}", key="manual_order")
        
        if st.button("Сгенерировать ТКП", type="primary", use_container_width=True, key="manual_button"):
            if not uploaded_files:
                st.error("Сначала загрузите CSV или PDF файлы с товарами")
            
            if not customer_query.strip():
                st.error("Введите запрос")
            
            with st.spinner("Поиск подходящих товаров..."):
                matches = ai_search.find_products(customer_query)
            
            if not matches:
                st.error("Не найдено подходящих товаров. Попробуйте уточнить запрос.")
            
            st.write(f"Найдено товаров: {len(matches)}")
            
            with st.spinner("Создание документа..."):
                order_data = {'order_id': order_number}
                doc, total_cost = doc_generator.generate_tcp_document(order_data, matches)
                
                doc_buffer = io.BytesIO()
                doc.save(doc_buffer)
                doc_buffer.seek(0)
            
            st.success(f"ТКП готов! Стоимость: {total_cost} руб.")
            
            st.download_button(
                label="Скачать ТКП (tcp.docx)",
                data=doc_buffer,
                file_name="tcp.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary",
                use_container_width=True
            )
    
    with tab2:
        st.header("2. Обработка JSON запросов")
        st.info("Загрузите JSON файл с запросами. Формат JSON должен содержать поле 'query' с текстом запроса.")
        
        uploaded_json_files = st.file_uploader(
            "Выберите JSON файлы с запросами",
            type=['json'],
            accept_multiple_files=True,
            key="json_uploader"
        )
        
        if uploaded_json_files:
            if not uploaded_files:
                st.error("Сначала загрузите каталог товаров (CSV/PDF файлы)")
            else:
                if st.button("Обработать JSON запросы", type="primary", use_container_width=True, key="json_button"):
                    all_results = []
                    
                    for json_file in uploaded_json_files:
                        st.write(f"Обработка файла: {json_file.name}")
                        results = json_processor.process_json_file(json_file)
                        all_results.extend(results)
                    
                    if all_results:
                        st.success(f"Обработано запросов: {len(all_results)}")
                        
                        # Показываем результаты для каждого запроса
                        for result in all_results:
                            with st.expander(f"Запрос {result['query_id']}: {result['query_text']}"):
                                st.write(f"Найдено товаров: {len(result['matches'])}")
                                st.write(f"Общая стоимость: {result['total_cost']} руб.")
                                
                                st.download_button(
                                    label=f"Скачать ТКП для запроса {result['query_id']}",
                                    data=result['document'],
                                    file_name=result['filename'],
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                    key=f"download_{result['query_id']}"
                                )
                    else:
                        st.error("Не удалось обработать ни одного запроса из JSON файлов")

if __name__ == "__main__":
    main()