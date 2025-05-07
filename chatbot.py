import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import datetime
import requests

# --- КОНФИГУРАЦИЯ ЧАТ-БОТА ---
PINECONE_API_KEY = "pcsk_zKtru_Ti1s6iQDrrDoqK1s4HKY4QjkDqf5KrCGvGnMgkqsYq3rffYWqV2FpAWwXqxCzkQ"  
PINECONE_INDEX_NAME = "game-bugs-index"
# Идентификатор модели должен быть ТОЧНО таким же, как в populate_pinecone.py
EMBEDDING_MODEL_ID = 'intfloat/multilingual-e5-large'
# Порог схожести для определения, является ли найденный баг релевантным
SIMILARITY_THRESHOLD = 0.8
N8N_WEBHOOK_URL = "https://tvirabyan.app.n8n.cloud/webhook/71abe225-0253-4899-849a-f0522f3b9603" 

class GameBugChatbot:
    def __init__(self):
        self.pinecone_api_key = PINECONE_API_KEY
        self.pinecone_index_name = PINECONE_INDEX_NAME
        self.embedding_model_id = EMBEDDING_MODEL_ID
        self.similarity_threshold = SIMILARITY_THRESHOLD
        
        self.embedding_model = None
        self.pinecone_index = None
        self.pc_client = None

    def _prepare_and_send_init_error_log(self, error_message, query_context="-"):
        """Вспомогательный метод для подготовки и отправки лога ошибки инициализации."""
        log_payload = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "user_query_context": query_context, # Контекст запроса здесь обычно отсутствует
            "bot_response_text": error_message, # Сообщение об ошибке
            "found_bug_id": None,
            "pinecone_score": None,
            "search_status": "Критическая ошибка инициализации" # Четкий статус для n8n
        }
        self.log_to_n8n(log_payload)

    def initialize_resources(self):
        """Загружает модель эмбеддингов и подключается к Pinecone."""
        print(f"Загрузка модели SentenceTransformer ({self.embedding_model_id})...")
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_id)
            print("Модель эмбеддингов успешно загружена.")
        except Exception as e:
            error_msg = f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить модель эмбеддингов: {e}"
            print(error_msg)
            self._prepare_and_send_init_error_log(error_msg)
            return False

        print("Инициализация клиента Pinecone...")
        try:
            self.pc_client = Pinecone(api_key=self.pinecone_api_key)
            print(f"Подключение к индексу '{self.pinecone_index_name}'...")
            if self.pinecone_index_name not in [idx.name for idx in self.pc_client.list_indexes().indexes]:
                error_msg = f"КРИТИЧЕСКАЯ ОШИБКА: Индекс '{self.pinecone_index_name}' не найден в Pinecone."
                print(error_msg)
                print("Пожалуйста, убедитесь, что скрипт populate_pinecone.py был успешно выполнен.")
                self._prepare_and_send_init_error_log(error_msg)
                return False
            self.pinecone_index = self.pc_client.Index(self.pinecone_index_name)
            print(f"Успешно подключено к индексу. Статистика: {self.pinecone_index.describe_index_stats()}")
            return True
        except Exception as e:
            error_msg = f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось подключиться к Pinecone: {e}"
            print(error_msg)
            self._prepare_and_send_init_error_log(error_msg)
            return False

    def get_bot_response(self, user_query):
        """Обрабатывает запрос пользователя, ищет в Pinecone и возвращает ответ."""
        if not self.embedding_model or not self.pinecone_index:
            return "Бот не инициализирован должным образом.", None, 0.0, "Ошибка инициализации"

        try:
            print(f"Векторизация запроса: '{user_query}'")
            query_vector = self.embedding_model.encode(user_query).tolist()
        except Exception as e:
            print(f"Ошибка при векторизации запроса: {e}")
            return "Произошла ошибка при обработке вашего запроса.", None, 0.0, "Ошибка векторизации"

        bot_response_text = "Не знаю"
        found_bug_id = None
        pinecone_score = 0.0
        search_status = "Не найден"

        try:
            print("Поиск в Pinecone...")
            query_results = self.pinecone_index.query(
                vector=query_vector,
                top_k=1, # Ищем 1 самый похожий баг
                include_metadata=True
            )

            if query_results.matches:
                best_match = query_results.matches[0]
                pinecone_score = best_match.score
                found_bug_id = best_match.id
                
                print(f"Найден ближайший баг: ID={found_bug_id}, Схожесть={pinecone_score:.4f}")

                if pinecone_score >= self.similarity_threshold:
                    metadata = best_match.metadata
                    if metadata:
                        title = metadata.get('original_title', '[Нет заголовка]')
                        description = metadata.get('original_description', '[Нет описания]')
                        bot_response_text = f"Найден похожий баг (схожесть: {pinecone_score:.2f}):\nID: {found_bug_id}\nЗаголовок: {title}\nОписание: {description}"
                        search_status = "Найден"
                    else:
                        bot_response_text = f"Найден баг ID: {found_bug_id} (схожесть: {pinecone_score:.2f}), но метаданные отсутствуют."
                        search_status = "Найден, нет метаданных"
                else:
                    print(f"Схожесть ({pinecone_score:.4f}) ниже порога ({self.similarity_threshold}). Ответ: Не знаю.")
                    search_status = "Найден, схожесть низкая"
            else:
                print("Pinecone не вернул совпадений.")
        
        except Exception as e:
            print(f"Ошибка при запросе к Pinecone: {e}")
            bot_response_text = "Произошла ошибка при поиске информации о баге."
            search_status = "Ошибка Pinecone при поиске"
        
        return bot_response_text, found_bug_id, pinecone_score, search_status

    def log_to_n8n(self, log_data):
        """Отправляет данные лога в n8n."""
        if not N8N_WEBHOOK_URL:
            print("N8N_WEBHOOK_URL не настроен. Логирование в n8n пропускается.")
            print("\n--- Данные для n8n (локальный вывод) ---")
            for key, value in log_data.items():
                print(f"{key}: {value}")
            print("-------------------------------------")
            return

        try:
            print(f"Отправка данных в n8n на {N8N_WEBHOOK_URL}...")
            response = requests.post(N8N_WEBHOOK_URL, json=log_data, timeout=10)
            response.raise_for_status() 
            print(f"Данные успешно отправлены в n8n. Статус: {response.status_code}")
        except requests.exceptions.Timeout:
            print(f"ОШИБКА N8N LOG: Таймаут при отправке данных в n8n.")
        except requests.exceptions.ConnectionError:
            print(f"ОШИБКА N8N LOG: Ошибка соединения при отправке данных в n8n.")
        except requests.exceptions.HTTPError as e:
            print(f"ОШИБКА N8N LOG: HTTP ошибка при отправке данных в n8n: {e.response.status_code} - {e.response.text}")
        except requests.exceptions.RequestException as e:
            print(f"ОШИБКА N8N LOG: Общая ошибка при отправке данных в n8n: {e}")
        except Exception as e:
            print(f"ОШИБКА N8N LOG: Непредвиденная ошибка при отправке логов: {e}")

    def run(self):
        """Основной цикл работы чат-бота."""
        if not self.initialize_resources():
            print("Завершение работы бота из-за ошибок инициализации.")
            return

        print("\nЧат-бот информации о багах запущен!")
        print("Введите описание проблемы или 'выход' для завершения.")

        while True:
            user_input = input("\nВы: ").strip()
            if user_input.lower() in ["выход", "exit", "quit", "пока"]:
                print("До свидания!")
                break
            
            if not user_input:
                continue

            bot_answer, bug_id, score, status = self.get_bot_response(user_input)
            print(f"Бот: {bot_answer}")

            # Подготовка данных для логирования в n8n
            log_payload = {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "user_query": user_input,
                "bot_response_text": bot_answer, # Текстовый ответ бота
                "found_bug_id": bug_id,
                "pinecone_score": float(score) if score is not None else None,
                "search_status": status
            }
            self.log_to_n8n(log_payload)

if __name__ == "__main__":
    chatbot = GameBugChatbot()
    chatbot.run() 