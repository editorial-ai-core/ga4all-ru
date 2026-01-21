# GA4 Streamlit Analytics

Универсальный Streamlit-инструмент для выгрузки данных из **Google Analytics 4**  
через **Service Account** (Analytics Data API).

Работает без OAuth, браузерных логинов и локальных файлов ключей.  
Подходит для быстрого запуска новых Streamlit-приложений.

---

## Возможности

- Выгрузка метрик GA4 по списку URL / pagePath
- Поддержка:
  - `https://site/path`
  - `site/path`
  - `/path`
- Автоматическая нормализация URL
- Топ материалов по просмотрам
- Общие показатели сайта (sessions, users, views)
- Экспорт в CSV

---

## Требования

- Python 3.10+
- Доступ к GA4 Property
- Service Account, добавленный в GA4 с ролью **Analyst**

---

## Установка

```bash
pip install -r requirements.txt
