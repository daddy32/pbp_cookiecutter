import xml.etree.ElementTree as et 

def parseElement(recordList, indent = 0):
    print(' '*indent + '{}'.format(recordList.tag))
    for child in recordList:
        #print('    ' + child.tag)
        parseElement(child, indent + 4)
       

# XML to pandas inspired by:
# https://robertopreste.com/blog/parse-xml-into-dataframe
       
def xmlElementToDf(
        xmlRoot: et.Element, 
        columns: list, 
        namespace = ''
    ) -> pd.DataFrame:
    '''
        Parses all first-level children of given XML element, extracts values
        from their first-level childred given by names and transforms them
        into rows and columns of a Pandas DataFrame
    '''
    
    # TODO: Detect column names automatically, remove namespace from them for df.
    out_df = pd.DataFrame(columns = columns)
    for node in xmlRoot: 
        values = []
        for column in df_cols:
            cell = node.find(namespace + column)
            val = cell.text if cell is not None else None
            values.append(val)

        out_df = out_df.append(
            pd.Series(values,index = df_cols), 
           ignore_index = True
        )
        
    return out_df
