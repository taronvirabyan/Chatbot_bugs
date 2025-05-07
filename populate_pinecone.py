import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# --- КОНФИГУРАЦИЯ ---
PINECONE_API_KEY = "pcsk_zKtru_Ti1s6iQDrrDoqK1s4HKY4QjkDqf5KrCGvGnMgkqsYq3rffYWqV2FpAWwXqxCzkQ" 
PINECONE_INDEX_NAME = "game-bugs-index"

# Список багов из вашего ТЗ
bugs_data = [
    {
        "id": "bug_1",
        "title": "Ошибка при загрузке уровня в многопользовательском режиме. После загрузки уровня клиент зависает.",
        "description": "При загрузке уровня в многопользовательском режиме клиент зависает сразу после завершения процесса загрузки. Игрок не может продолжить игру, управление становится недоступным, и требуется принудительная перезагрузка клиента. Эта ошибка прерывает игровой процесс для всех участников сессии и делает невозможным проведение многопользовательских матчей. Предполагается, что проблема связана с некорректной синхронизацией данных или сетевыми задержками на этапе загрузки уровня."
    },
    {
        "id": "bug_2",
        "title": "Некорректное отображение текста в диалоговых окнах на некоторых разрешениях экрана.",
        "description": "На некоторых разрешениях экрана текст в диалоговых окнах отображается неправильно: строки текста могут обрываться, накладываться друг на друга или выходить за границы выделенных областей. Это приводит к тому, что часть информации становится нечитабельной или вовсе пропадает из поля зрения пользователя. Причиной может быть отсутствие адаптивной верстки интерфейса, который должен автоматически подстраиваться под разные размеры экранов."
    },
    {
        "id": "bug_3",
        "title": "При использовании оружия \"Автоган\" текстуры объекта исчезают, при этом остаются слышны звуки выстрелов.",
        "description": "При использовании оружия \"Автомат\" происходит исчезновение текстур самого оружия. Игрок всё ещё слышит звуки выстрелов, но на экране отсутствует визуальная модель оружия, что нарушает восприятие игрового процесса и может вызывать путаницу. Вероятной причиной является ошибка в анимации или рендеринге модели, происходящая в момент активации режима стрельбы."
    },
    {
        "id": "bug_4",
        "title": "При нажатии на кнопку \"Сохранить\" в разделе настроек профиля, изменения не сохраняются.",
        "description": "В разделе настроек профиля после изменения параметров и нажатия кнопки \"Сохранить\" изменения не фиксируются. После перезагрузки страницы или повторного входа в профиль все параметры сбрасываются к предыдущим значениям. Это создает ощущение ненадежной работы системы и может вызывать разочарование у пользователей. Скорее всего, ошибка возникает на этапе отправки или сохранения данных на сервере."
    },
    {
        "id": "bug_5",
        "title": "При попытке загрузить изображение в профиль, система не обрабатывает файлы.",
        "description": "При попытке загрузить новое изображение в профиль система не обрабатывает загружаемый файл. Изображение не появляется, при этом пользователю не всегда предоставляется информация о причине неудачи. Такая проблема мешает персонализации профиля и портит пользовательский опыт. Возможной причиной является сбой при валидации файла или ошибка передачи данных на сервер для обработки изображения."
    }
]

def main():
    # Загружаем модель для создания эмбеддингов
    # Модель будет загружена при первом вызове, это может занять некоторое время.
    print("Загрузка модели SentenceTransformer (intfloat/multilingual-e5-large)...")
    try:
        embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
        print("Модель успешно загружена.")
    except Exception as e:
        print(f"Ошибка при загрузке модели SentenceTransformer: {e}")
        return

    print("Инициализация Pinecone клиента...")
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
    except Exception as e:
        print(f"Ошибка при инициализации Pinecone: {e}")
        return

    print(f"Подключение к индексу '{PINECONE_INDEX_NAME}'...")
    try:
        # Убедимся, что индекс существует (хотя мы его уже создали через UI)
        indexes_list_response = pc.list_indexes()
        index_names = []
        if indexes_list_response and hasattr(indexes_list_response, 'indexes'):
            index_names = [idx_spec.name for idx_spec in indexes_list_response.indexes]
        
        if PINECONE_INDEX_NAME not in index_names:
            print(f"Индекс '{PINECONE_INDEX_NAME}' не найден. Пожалуйста, создайте его в Pinecone консоли.")
            return

        index = pc.Index(PINECONE_INDEX_NAME)
        print(f"Успешно подключено к индексу. Статистика: {index.describe_index_stats()}")

    except Exception as e:
        print(f"Ошибка при подключении к индексу: {e}")
        return

    print("Подготовка данных для загрузки...")
    vectors_to_upsert = []
    for bug in bugs_data:
        # Формируем текст, который будет векторизован Pinecone

        full_bug_text_for_embedding = f"Заголовок: {bug['title']}. Описание: {bug['description']}"

        print(f"Векторизация бага: {bug['id']}...")
        try:
            vector = embedding_model.encode(full_bug_text_for_embedding).tolist()
        except Exception as e:
            print(f"Ошибка при векторизации бага {bug['id']}: {e}")
            continue 

        vector_data = {
            "id": bug["id"],
            "values": vector, 
            "metadata": {
                "original_title": bug["title"],
                "original_description": bug["description"],
                "source_text_for_embedding": full_bug_text_for_embedding
            }
        }

        vectors_to_upsert.append(vector_data)
        print(f"Подготовлен баг: {bug['id']}")

    if not vectors_to_upsert:
        print("Нет данных для загрузки.")
        return

    print(f"\nЗагрузка {len(vectors_to_upsert)} векторов в Pinecone...")
    try:
        index.upsert(vectors=vectors_to_upsert)
        print("Данные успешно загружены!")
        print(f"Новая статистика индекса: {index.describe_index_stats()}")
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")

if __name__ == "__main__":
    main()
