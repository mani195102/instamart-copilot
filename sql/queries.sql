-- Instamart Customer Intelligence -- Part A SQL analysis (SQLite 3.25+)
-- Each query starts with a "-- name: Qn" marker; src/part_a_analysis.py splits on these markers.
-- Unless stated otherwise only completed orders count as revenue or spend. All amounts are INR.

-- name: Q1
-- WHERE filters rows before grouping; HAVING filters groups after aggregation.
SELECT customer_id,
       COUNT(*)                    AS big_orders,
       ROUND(SUM(order_amount), 2) AS total_big_order_inr
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_big_order_inr DESC;

-- name: Q2
-- Revenue by city and product category (three-table join).
SELECT c.city,
       p.category,
       ROUND(SUM(o.order_amount), 2) AS revenue_inr
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN products  p ON p.product_id  = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue_inr DESC;

-- name: Q3
-- Products priced above the average catalogue price (scalar subquery).
SELECT product_id, product_name, category, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- name: Q4
-- Customers with no completed order (LEFT JOIN ... IS NULL).
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o
       ON o.customer_id = c.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- name: Q5
-- Customers whose total completed spend is above the average customer total (two CTEs).
WITH customer_totals AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_total AS (
    SELECT AVG(total_spend) AS avg_spend FROM customer_totals
)
SELECT t.customer_id,
       ROUND(t.total_spend, 2) AS total_spend_inr,
       ROUND(a.avg_spend, 2)   AS avg_spend_inr
FROM customer_totals t
CROSS JOIN avg_total a
WHERE t.total_spend > a.avg_spend
ORDER BY t.total_spend DESC;

-- name: Q6
-- Ranking window functions within each city.
WITH spend AS (
    SELECT o.customer_id, c.city, ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM orders o
    JOIN customers c ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY o.customer_id, c.city
)
SELECT city, customer_id, total_spend,
       RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM spend
ORDER BY city, row_num;

-- name: Q7
-- Daily completed revenue with a running total.
WITH daily AS (
    SELECT SUBSTR(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT day,
       ROUND(revenue, 2)                             AS revenue_inr,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2)    AS cumulative_revenue_inr
FROM daily
ORDER BY day;
