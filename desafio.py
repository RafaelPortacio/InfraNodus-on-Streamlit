from auxiliar_functions import *
import streamlit as st
import pytesseract
import streamlit.components.v1 as components
from pdf2image import convert_from_bytes
from pyvis.network import Network


uploaded_files = st.file_uploader(label="Choose a PDF file",
                                  accept_multiple_files=True,
                                  type=['pdf'])
                                  

for uploaded_file in uploaded_files:
    # Getting data
    bytes_data = uploaded_file.read()
    st.write("Arquivo:", uploaded_file.name)


    #------------------------
    # Step 0: Converting data
    #------------------------
    # From bytes to image
    pages_images = convert_from_bytes(bytes_data)
    
    # From image to text
    text = ''
    for img in pages_images:
        text += '\n' + str(((pytesseract.image_to_string(img))))

    # Separating paragraphs
    paragraphs = text.split('\n\n')
    

    #---------------------------
    # Step 1: Text Normalization
    #---------------------------
    paragraphs = apply(paragraphs,
                       remove_punctuation,
                       lowercase,
                       convert2lemma)

                            
    #---------------------------                        
    # Step 2: Stop words Removal
    #---------------------------
    stop_list = stopwords.words('english')
    
    remove_stopwords = lambda text: remove_stpwrds(text,
                                                   stop_list)
    paragraphs = apply(paragraphs,
                       remove_stopwords)

    
    #-----------------------------------
    # Step 3: Text-to-Network Conversion
    #-----------------------------------
    text_graph = graphfy(paragraphs)

    
    #------------------------------------
    # Step 4: Extracting Most Influential 
    # Keywords Using Betweenness Centrality
    #--------------------------------------
    text_graph,centrality_dict = influential_nodes(text_graph, 200)
    
    max_cen = max(list(centrality_dict.values()))
    
    
    #----------------------------------------
    # Step 5: Topic Modelling Using Community 
    # Detection and Force-Atlas Layout
    #---------------------------------
    text_graph, communities = nodes_communities(text_graph)
    
    # Adjusting the plot
    text_net = Network(height='435px',
                       width='675px',
                       bgcolor='#0e1117',
                       font_color='white')
    
    calc_size = lambda x: size_from_centrality(x, max_cen)
    
    
    
    text_net.from_nx(text_graph,
                     node_size_transf=calc_size)
    
    nx.set_node_attributes(text_graph,
                       centrality_dict, "size")
                       
    
    for node in text_net.nodes:
        # tamanho dos nós
        size = str(12+node['size']/4)
        node['font'] = size+'px arial white'
        node['labelHighlightBold'] = True
    
    
    text_net.force_atlas_2based(gravity=-15,
                                central_gravity=0.001,
                                spring_length=10,
                                spring_strength=0.008,
                                damping=1,
                                overlap=1)
    
    text_net.save_graph('pyvis_graph.html')
    HtmlFile = open('pyvis_graph.html', 'r',
                    encoding='utf-8')
    
    #--------------------------------
    # Step 7: Discourse Structure and
    # the Measure of Discourse Bias
    #------------------------------
    modular = modularity(text_graph, communities)
    
    top_cluster_perc = top_cluster_percent(communities)
    calculated_entropy = graph_entropy(text_graph)
    classif = classification(modular,
                             top_cluster_perc,
                             calculated_entropy)
    
    st.write('Modularity: '+str(modular))
    st.write('Top Cluster Percentage: '+str(top_cluster_perc))
    st.write('''Top 4's entropy: '''+str(calculated_entropy))
    st.write('Classification: '+classif)
    st.write('Graph:')
    # Load HTML file in HTML component for display on Streamlit page
    components.html(HtmlFile.read(),
                    scrolling=True,
                    height=435)