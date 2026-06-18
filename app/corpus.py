"""Synthetic Northwind Support knowledge base.

These are the source documents the RAG slice ingests. They are intentionally
small, factual, and self-contained so that the golden eval set can reference
exact facts. `write_corpus()` materialises them as markdown files under
`app/data/corpus/` (one file per doc id).

Doc ids are stable identifiers used as citations and as `source_docs` in the
golden set, so DO NOT rename them without updating eval/golden_set.jsonl.
"""
from __future__ import annotations

from .config import CORPUS_DIR

DOCS: dict[str, str] = {
    "shipping-domestic": """# Domestic Shipping

Northwind ships domestically via standard and expedited options.

- Standard shipping takes 5 to 7 business days.
- Standard shipping is free on orders over $50; otherwise it costs $5.99.
- Expedited shipping takes 2 to 3 business days and costs $14.99 flat.
- Orders placed before 1pm ET on a business day ship the same day.
- We ship Monday through Friday, excluding public holidays.
""",
    "shipping-international": """# International Shipping

Northwind ships to 40 countries.

- International delivery takes 10 to 21 business days.
- Import duties, customs fees, and taxes are the responsibility of the customer.
- International orders are not eligible for free shipping.
- Tracking is available but may not update while in transit through customs.
""",
    "returns-policy": """# Returns Policy

- Items can be returned within 30 days of delivery.
- Returned items must be unused and have original tags attached.
- We provide a prepaid return shipping label for domestic returns.
- Final-sale and clearance items cannot be returned.
- To start a return, go to Orders in your account and select Return Item.
""",
    "refunds-processing": """# Refund Processing

- Refunds are issued after we receive and inspect the returned item.
- Once approved, refunds take 5 to 10 business days to appear.
- Refunds are always issued to the original payment method.
- If you paid with a Northwind gift card, the refund is returned as gift-card balance.
- Original shipping charges are non-refundable unless the item was defective.
""",
    "order-tracking": """# Order Tracking

- A tracking number is emailed to you as soon as your order ships.
- You can also track orders under the Orders tab in your account.
- Tracking can take up to 24 hours to show movement after it is generated.
""",
    "order-cancellation": """# Order Cancellation

- You can cancel an order yourself within 1 hour of placing it, from the Orders tab.
- After 1 hour, contact support and we will try to stop it, but cannot guarantee it.
- Once an order has shipped it cannot be cancelled; you must return it instead.
""",
    "order-modification": """# Changing an Order

- You can change the shipping address within 1 hour of placing an order.
- Items in an order cannot be added or swapped; cancel and reorder instead.
- Orders that have already shipped cannot be modified.
""",
    "payment-methods": """# Accepted Payment Methods

- We accept Visa, Mastercard, and American Express.
- We accept PayPal and Northwind gift cards.
- We do not accept cash on delivery (COD) or cryptocurrency.
- Only one gift card can be applied per order.
""",
    "payment-failures": """# Declined or Failed Payments

- If a card is declined, verify the card number, expiry, CVV, and billing ZIP.
- Ensure sufficient funds and that the card is approved for online purchases.
- After 3 failed attempts, checkout is locked for 24 hours for security.
- Contact your bank if the card keeps getting declined.
""",
    "account-creation": """# Creating an Account

- Sign up with an email address and a password of at least 8 characters.
- A verification link is emailed to you; click it to activate the account.
- The verification link expires after 24 hours.
""",
    "account-password-reset": """# Resetting Your Password

- Use the Forgot Password link on the sign-in page.
- A reset email is sent; the reset link is valid for 60 minutes.
- If you do not receive it, check spam or request a new link.
""",
    "subscription-plans": """# Subscription Plans

- Basic is free and includes standard features.
- Plus costs $9 per month.
- Pro costs $19 per month and adds priority support and early access to new features.
- Annual billing saves two months compared with monthly billing.
""",
    "subscription-cancellation": """# Cancelling a Subscription

- You can cancel anytime from Settings > Subscription.
- You keep access until the end of the current billing period.
- We do not issue prorated refunds for partial billing periods.
""",
    "warranty-policy": """# Warranty

- Electronics come with a 1-year limited warranty.
- The warranty covers manufacturing defects, not accidental or misuse damage.
- Warranty claims require proof of purchase.
""",
    "damaged-items": """# Damaged or Defective Items

- Report a damaged or defective item within 48 hours of delivery.
- Include photos of the damage when you contact support.
- Approved claims receive a free replacement at no shipping cost.
""",
    "product-availability": """# Product Availability

- Out-of-stock items display an expected restock date when available.
- You can sign up for a back-in-stock alert on the product page.
- Adding an item to your cart does not reserve stock.
""",
    "discounts-promotions": """# Discounts and Promotions

- Only one promo code can be applied per order.
- Promo codes cannot be combined with each other.
- Subscribing to the newsletter gives you a one-time 10% off code.
""",
    "data-privacy": """# Data and Privacy

- We do not sell your personal data to third parties.
- You can request deletion of your account data by emailing privacy@northwind.example.
- We honor GDPR and CCPA data-access and deletion requests.
""",
    "contact-support": """# Contacting Support

- Live chat is available 24/7 from the Help menu.
- Email support@northwind.example for a response within 24 hours.
- Phone support is available Monday to Friday, 9am to 6pm ET.
""",
    "business-accounts": """# Business Accounts

- Orders over 50 units qualify for volume discounts.
- Business accounts get a dedicated account manager.
- Approved business accounts can use Net-30 payment terms.
""",
}


def write_corpus() -> list[str]:
    """Write each doc to app/data/corpus/<doc_id>.md. Returns the file paths."""
    paths: list[str] = []
    for doc_id, body in DOCS.items():
        path = CORPUS_DIR / f"{doc_id}.md"
        path.write_text(body, encoding="utf-8")
        paths.append(str(path))
    return paths


if __name__ == "__main__":
    written = write_corpus()
    print(f"Wrote {len(written)} corpus documents to {CORPUS_DIR}")
 