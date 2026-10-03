-- E-Commerce Hackathon: Task B business queries
-- Database: ecommerce_clean.db
-- Net revenue = quantity * unit_price * (1 - discount), non-returned orders only

-- 1. Total net revenue
SELECT
    ROUND(SUM(quantity * unit_price), 2)                                       AS gross_revenue,
    ROUND(SUM(quantity * unit_price * discount), 2)                            AS discount_amount,
    ROUND(SUM(CASE WHEN returned = 1
                   THEN quantity * unit_price * (1 - discount) ELSE 0 END), 2) AS returned_amount,
    ROUND(SUM(CASE WHEN returned = 0
                   THEN quantity * unit_price * (1 - discount) ELSE 0 END), 2) AS net_revenue
FROM orders;

-- 2. Top 10 customers by total spending
SELECT
    c.customer_id,
    c.customer_name,
    c.city,
    COUNT(o.order_id)                                         AS number_of_orders,
    ROUND(SUM(o.quantity * o.unit_price * (1 - o.discount)), 2) AS total_spending
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.returned = 0
GROUP BY c.customer_id, c.customer_name, c.city
ORDER BY total_spending DESC
LIMIT 10;

-- 3. Category-wise revenue, order count and quantity sold
SELECT
    p.category,
    ROUND(SUM(o.quantity * o.unit_price * (1 - o.discount)), 2) AS net_revenue,
    COUNT(o.order_id)                                         AS order_count,
    SUM(o.quantity)                                           AS quantity_sold,
    ROUND(100.0 * SUM(o.quantity * o.unit_price * (1 - o.discount))
          / SUM(SUM(o.quantity * o.unit_price * (1 - o.discount))) OVER (), 2) AS revenue_share_pct
FROM orders o
JOIN products p ON o.product_id = p.product_id
WHERE o.returned = 0
GROUP BY p.category
ORDER BY net_revenue DESC;

-- 4. Monthly net revenue trend
WITH monthly AS (
    SELECT
        strftime('%Y-%m', order_date)                       AS month,
        ROUND(SUM(quantity * unit_price * (1 - discount)), 2) AS net_revenue,
        COUNT(order_id)                                     AS order_count
    FROM orders
    WHERE returned = 0
    GROUP BY month
)
SELECT
    month,
    net_revenue,
    order_count,
    ROUND(100.0 * (net_revenue - LAG(net_revenue) OVER (ORDER BY month))
          / LAG(net_revenue) OVER (ORDER BY month), 2) AS mom_growth_pct
FROM monthly
ORDER BY month;

-- 5. Top 5 products by net revenue
SELECT
    p.product_id,
    p.product_name,
    p.category,
    p.brand,
    SUM(o.quantity)                                           AS units_sold,
    ROUND(SUM(o.quantity * o.unit_price * (1 - o.discount)), 2) AS net_revenue
FROM orders o
JOIN products p ON o.product_id = p.product_id
WHERE o.returned = 0
GROUP BY p.product_id, p.product_name, p.category, p.brand
ORDER BY net_revenue DESC
LIMIT 5;

