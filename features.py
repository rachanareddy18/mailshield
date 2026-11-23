"""Text cleaning, phishing red flags, sender reputation, spam type and 'Email DNA' scores."""
import re

LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
URL_RE = re.compile(r"(https?://[^\s<>\"']+|www\.[^\s<>\"']+)", re.I)
MAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
TAG_RE = re.compile(r"<[^>]+>")
MIXED_RE = re.compile(r"\b(?=\w*[a-zA-Z])(?=\w*\d)\w+\b")
LEET_INSIDE = re.compile(r"[a-z][01345][a-z]", re.I)


def clean_text(text) -> str:
    """Normalise an email so tricks like 'fr33 pr1ze' become normal words."""
    t = str(text)[:5000].lower()
    t = TAG_RE.sub(" ", t)
    t = URL_RE.sub(" urltoken ", t)
    t = MAIL_RE.sub(" emailtoken ", t)
    t = MIXED_RE.sub(lambda m: m.group().translate(LEET), t)
    t = re.sub(r"\d+", " numtoken ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


URGENT = ["urgent", "immediately", "act now", "within 24 hours", "last chance", "expires", "final notice",
          "account suspended", "verify now", "limited time", "hurry", "today only", "reply fast", "asap"]
MONEY = ["winner", "you won", "prize", "lottery", "free gift", "cash", "million", "claim", "congratulations",
         "guaranteed", "earn", "work from home", "loan", "discount", "% off", "cashback", "voucher", "refund",
         "bitcoin", "invest"]
SENSITIVE = ["password", "otp", "cvv", "pin", "bank account", "credit card", "debit card", "aadhaar", "ssn",
             "social security", "card details", "login details", "identity", "upi"]
REQUEST_VERBS = r"\b(send|enter|reply|provide|share your|submit|confirm|verify|update|type|give|click)\b"
SHORTENERS = ["bit.ly", "tinyurl", "goo.gl", "t.co", "cutt.ly", "is.gd", "rb.gy", "shorturl"]
BAD_EXT = [".exe", ".scr", ".zip", ".bat", ".apk", ".vbs", "enable macros"]
BAD_TLDS = {"xyz", "top", "ru", "tk", "click", "loan", "gq", "ml", "cf", "icu", "buzz", "work", "site"}
FREE_MAIL = {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "rediffmail.com", "proton.me", "icloud.com"}
TRUSTED_SUFFIX = (".edu", ".ac.in", ".edu.in", ".gov", ".gov.in")
BRANDS = ["paypal", "amazon", "google", "microsoft", "apple", "netflix", "sbi", "hdfc", "icici", "paytm", "flipkart",
          "irctc", "linkedin", "facebook", "instagram", "whatsapp", "tcs", "infosys", "dhl", "gmail", "irs", "upi"]
SEC_WORDS = ["secure", "verify", "login", "update", "support", "alert", "billing", "account", "helpdesk", "refund"]


def _host(url):
    return re.sub(r"^https?://", "", url.lower()).split("/")[0].split(":")[0]


def _core(domain):
    labels = domain.split(".")
    return labels[-2] if len(labels) >= 2 else labels[0]


def red_flags(text):
    """Rule-based phishing checks -> list of (name, detail, weight)."""
    text, flags = str(text), []
    low = text.lower()
    for u in URL_RE.findall(text):
        host = _host(u.rstrip(".,);"))
        core = _core(host)
        if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host):
            flags.append(("IP-address link", host, 0.10))
        elif any(s in host for s in SHORTENERS):
            flags.append(("Shortened link", host, 0.10))
        elif host.split(".")[-1] in BAD_TLDS:
            flags.append(("Risky link ending", host, 0.08))
        elif any(b in host and core != b for b in BRANDS):
            flags.append(("Link imitates a brand", host, 0.10))
        elif LEET_INSIDE.search(core) or host.count("-") >= 2:
            flags.append(("Look-alike domain", host, 0.08))
    for m in re.finditer(r"<a[^>]*href=[\"']([^\"']+)[\"'][^>]*>([^<]+)</a>", text, re.I):
        href, label = m.group(1).lower(), m.group(2).lower().strip()
        if "." in label and label not in href:
            flags.append(("Link text differs from real URL", f"{label} -> {href}", 0.10))
    hit = [w for w in URGENT if w in low]
    if hit:
        flags.append(("Urgency pressure", ", ".join(hit[:3]), 0.04))
    hit = [w for w in MONEY if w in low]
    if hit:
        flags.append(("Money / prize bait", ", ".join(hit[:3]), 0.04))
    hit = [w for w in SENSITIVE if re.search(r"\b" + re.escape(w) + r"\b", low)]
    if hit and re.search(REQUEST_VERBS, low) and "do not share" not in low and "never ask" not in low:
        flags.append(("Asks for sensitive info", ", ".join(hit[:3]), 0.10))
    letters = [c for c in text if c.isalpha()]
    if len(letters) > 20 and sum(c.isupper() for c in letters) / len(letters) > 0.35:
        flags.append(("Too many CAPITALS", "", 0.04))
    if text.count("!") >= 3:
        flags.append(("Excess exclamation marks", "", 0.04))
    ext = [e for e in BAD_EXT if e in low]
    if ext:
        flags.append(("Risky attachment", ", ".join(ext), 0.10))
    return flags


def sender_analysis(sender):
    """Sender reputation: returns (score 0-1 or None, [(reason, detail)])."""
    s = (sender or "").strip().lower()
    if not s:
        return None, []
    m = re.fullmatch(r"(?:.*<)?([\w.+-]+)@([\w.-]+\.[a-z]{2,})>?", s)
    if not m:
        return 0.3, [("Not a valid email address", s)]
    local, domain = m.group(1), m.group(2)
    if domain.endswith(TRUSTED_SUFFIX):
        return 0.0, [("Education / government domain", domain)]
    labels, core, score, why = domain.split("."), _core(domain), 0.0, []
    if labels[-1] in BAD_TLDS:
        score += 0.35
        why.append(("Risky domain ending", "." + labels[-1]))
    if LEET_INSIDE.search(core) or re.search(r"\d{3,}", core):
        score += 0.30
        why.append(("Digits mixed into name (look-alike)", core))
    norm = domain.translate(LEET)
    for b in BRANDS:
        if (b in norm and _core(norm) != b) and domain not in FREE_MAIL:
            score += 0.40
            why.append(("Imitates a known brand", f"'{b}' inside {domain}"))
            break
    if domain in FREE_MAIL:
        why.append(("Free mail provider (neutral)", domain))
        if any(b in local for b in BRANDS) or any(w in local for w in ("support", "security", "helpdesk", "alert")):
            score += 0.30
            why.append(("Company-style name on a free address", local))
    elif any(w in domain for w in SEC_WORDS):
        score += 0.25
        why.append(("Security-sounding words in domain", domain))
    if domain.count(".") >= 3 or core.count("-") >= 2:
        score += 0.15
        why.append(("Long / hyphenated domain", domain))
    if re.search(r"\d{5,}", local):
        score += 0.15
        why.append(("Long number in address", local))
    if not why:
        why.append(("No suspicious pattern found", domain))
    return min(1.0, score), why


CATS = {
    "Phishing (account / credential)": ["verify", "password", "otp", "login", "suspended", "blocked", "confirm your identity",
                                         "pin", "cvv", "bank", "kyc", "unusual activity", "storage"],
    "Prize / lottery scam": ["won", "winner", "prize", "lottery", "congratulations", "gift card", "free gift", "lucky", "voucher"],
    "Fake job / internship": ["internship", "job", "salary", "registration fee", "visa", "offer letter", "hiring", "joining", "certificate"],
    "Loan / investment scam": ["loan", "invest", "bitcoin", "guaranteed", "returns", "cash", "earn", "income", "processing fee"],
    "Scholarship scam": ["scholarship", "grant", "stipend", "aadhaar"],
    "Parcel / delivery scam": ["parcel", "package", "delivery", "customs", "courier", "redelivery"],
    "Promo / shopping spam": ["discount", "% off", "sale", "buy now", "order now", "free spins", "pills", "limited time"],
    "Malware / fake invoice": [".exe", ".zip", "macros", "invoice", "attachment"],
    "Tech-support / subscription scam": ["virus", "technical support", "subscription", "renewed", "helpline", "refund"],
    "Advance-fee / impersonation": ["transfer", "beneficiary", "late minister", "gift cards", "reimburse", "inheritance"],
}


def spam_category(text, flags):
    low, scores = str(text).lower(), {}
    for cat, words in CATS.items():
        scores[cat] = sum(w in low for w in words)
    names = [f[0] for f in flags]
    if any(n in names for n in ("IP-address link", "Shortened link", "Link imitates a brand", "Look-alike domain",
                                "Link text differs from real URL", "Asks for sensitive info")):
        scores["Phishing (account / credential)"] += 2
    if "Risky attachment" in names:
        scores["Malware / fake invoice"] += 3
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General spam"


def dna_scores(text, flags, sender_score, p):
    """Eight 0-1 traits that make up an email's 'DNA' fingerprint."""
    text = str(text)
    low = text.lower()
    names = [f[0] for f in flags]
    letters = [c for c in text if c.isalpha()]
    caps = sum(c.isupper() for c in letters) / len(letters) if letters else 0
    link_flags = {"IP-address link", "Shortened link", "Risky link ending", "Link imitates a brand", "Look-alike domain",
                  "Link text differs from real URL"}
    links = 1.0 if link_flags & set(names) else (0.3 if URL_RE.search(text) else 0.0)
    return {
        "Urgency": min(1.0, sum(w in low for w in URGENT) / 2),
        "Money bait": min(1.0, sum(w in low for w in MONEY) / 2),
        "Sensitive ask": 1.0 if "Asks for sensitive info" in names else 0.0,
        "Risky links": links,
        "Shouting": min(1.0, max(caps / 0.35, text.count("!") / 5)),
        "Obfuscation": min(1.0, len(LEET_INSIDE.findall(text)) / 2),
        "Sender risk": float(sender_score or 0.0),
        "ML spam score": float(p),
    }


def strip_links(t):
    return URL_RE.sub("", re.sub(r"<a[^>]*>([^<]*)</a>", r"\1", str(t)))


def strip_words(t, words):
    for w in words:
        t = re.sub(re.escape(w), "", t, flags=re.I)
    return re.sub(r"\s{2,}", " ", t)
