from sentence_transformers import SentenceTransformer, util

class SBERTModel:
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2'):
        # Initialize the SBERT model
        self.model = SentenceTransformer(model_name)
        
        # Define reference queries inside the class
        # Reference queries for SBERT matching
        self.reference_queries = [
            # Company Description
            "Provide a brief description of the company in less than 100 words",
            "What is a short description of the company?",
            "Summarize the company profile",
            "Give a quick overview of the company",

            # Products or Services
            "Does the company sell products, services, or both?",
            "What products or services does the company offer?",
            "Summarize what the company sells in 20 words or less",
            "What are the main products or services of the company?",

            # Manufacturer or Distributor
            "Is the company a manufacturer, distributor, or both?",
            "Does the company manufacture or distribute goods?",
            "What role does the company play in the supply chain?",

            # Industry Segment
            "What industry segment does the company operate in?",
            "Which industry does the company belong to?",
            "What sector does the company specialize in?",

            # Founding Year
            "When was the company founded?",
            "What year was the company established?",
            "In which year did the company start?",

            # Number of Employees
            "How many employees does the company have?",
            "What is the total number of employees in the company?",
            "How big is the company's workforce?",

            # Headquarters Location
            "Where is the company headquartered?",
            "What is the location of the company's headquarters?",
            "Where is the company's main office?",

            # Employee Contact Information
            "List names, emails, and phone numbers of any employees at the company",
            "Who are the key employees of the company, along with their contact details?",
            "Provide the names and contact information of employees at the company",

            # B2B or B2C
            "Does the company serve B2B, B2C, or both types of customers?",
            "Is the company focused on B2B, B2C, or both markets?",
            "What types of customers does the company serve?",

            # Customer Industry
            "What industries do the company's customers belong to?",
            "Highlight the industries that the company's customers are part of",
            "Which sectors do the company's clients come from?",

            # Common Queries
            "What does the company do?",
            "Where is the company located?",
            "What is the company's main business?",
            "What services does the company provide?",
            "What products does the company offer?",
            "What are the main offerings of the company?",
            "Where is the company situated?",





            # Consumer Products
            "What products does the company sell in Food & Beverages?",
            "Does the company sell packaged foods, beverages, or snacks?",
            "What health supplements or vitamins does the company offer?",
            "Does the company sell personal care items like skincare or cosmetics?",
            "Does the company offer home care products such as detergents and cleaning products?",
            "What apparel or luxury goods does the company sell?",
            "Does the company offer consumer electronics such as smartphones or home appliances?",
            "What types of furniture or home decor items does the company sell?",
            "What health and wellness products, like fitness equipment or dietary supplements, does the company offer?",

            # Consumer Services
            "Does the company offer hospitality services such as hotels or resorts?",
            "What types of health and wellness services does the company provide, like fitness centers or spas?",
            "Does the company offer entertainment services such as cinemas or theme parks?",
            "Does the company provide retail services such as e-commerce or grocery stores?",
            "What education services, such as tutoring or online courses, does the company offer?",
            "Does the company provide financial services like personal loans or insurance?",
            "What food and beverage services, such as catering or delivery, does the company provide?",

            # Industrials
            "Is the company involved in manufacturing automotive or aerospace products?",
            "Does the company provide construction services for residential or commercial projects?",
            "Is the company part of the energy sector, including oil, gas, or renewable energy?",
            "What transportation services, such as logistics or shipping, does the company offer?",
            "Does the company manufacture chemicals, such as agricultural or specialty chemicals?",
            "Is the company involved in metals and mining, such as iron or precious metals?",
            "What industrial equipment, such as robotics or HVAC systems, does the company manufacture?",

            # Business Services
            "What type of consulting services does the company offer, such as IT or management consulting?",
            "Does the company provide facilities management services like cleaning or security?",
            "What human resources services, such as recruitment or payroll, does the company offer?",
            "Does the company provide IT services such as cloud computing or cybersecurity?",
            "What marketing or advertising services does the company provide?",
            "Does the company offer legal or accounting services?",
            "Does the company provide outsourcing services like business process outsourcing or call centers?",

            # Technology
            "Does the company offer software development services, such as SaaS or mobile apps?",
            "What types of hardware does the company manufacture, such as semiconductors or networking equipment?",
            "Is the company involved in telecommunications services such as mobile networks or satellite communications?",
            "What e-commerce services does the company offer, such as online retail or delivery tech?",
            "Does the company provide fintech services, such as payment processing or digital wallets?",
            "Is the company involved in cloud computing services like IaaS or PaaS?",
            "What cybersecurity solutions, such as identity management or data protection, does the company provide?",

            # General Questions (applicable to all)
            "What does the company do?",
            "Where is the company located?",
            "Where is the company headquartered?",
            "When was the company founded?",
            "Is the company a manufacturer or distributor?",
            "What products or services does the company offer?",
            "Does the company serve B2B or B2C customers?",
            "What industries does the company operate in?",
            "Who are the key employees of the company, and how can they be contacted?",
            "What industries do the company's customers belong to?"
    ]



    def analyze_query(self, user_query, threshold=0.6):
        """
        Analyze the user query using SBERT embeddings and return the most relevant reference query
        if it meets the similarity threshold.
        """
        print("Using SBERT to analyze the query...")
        
        # Generate embeddings for the user query and reference queries
        user_embedding = self.model.encode(user_query, convert_to_tensor=True)
        reference_embeddings = self.model.encode(self.reference_queries, convert_to_tensor=True)

        # Calculate cosine similarity between user query and reference queries
        similarities = util.pytorch_cos_sim(user_embedding, reference_embeddings)

        # Find the best matching query
        best_match_idx = similarities.argmax()
        best_match_query = self.reference_queries[best_match_idx]
        best_match_score = similarities[0, best_match_idx].item()

        # Print SBERT similarity score
        print(f"SBERT similarity score: {best_match_score:.2f}")

        # Return best match if it exceeds the threshold, otherwise return the original query
        if best_match_score >= threshold:
            print(f"SBERT Match found: '{best_match_query}' with a score of {best_match_score:.2f}")
            return best_match_query  # Return the best matching query
        else:
            print("SBERT did not find a relevant match. Proceeding with the original query.")
            return user_query  # Return the original query if no match is found