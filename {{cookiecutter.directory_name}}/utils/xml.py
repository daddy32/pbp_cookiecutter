import xml.etree.ElementTree as et 
import pandas as pd
import collections

def parseElement(recordList, indent = 0):
    print(' '*indent + '{}'.format(recordList.tag))
    for child in recordList:
        #print('    ' + child.tag)
        parseElement(child, indent + 4)
       

# XML to pandas inspired by:
# https://robertopreste.com/blog/parse-xml-into-dataframe

# Mozno upravit columns na dict - path -> label

def xmlElementToDf(
        xmlRoot: et.Element, 
        columns, 
        namespace = ''
    ) -> pd.DataFrame:
    '''
        Parses all first-level children of given XML element, extracts values
        from their first-level childred given by names and transforms them
        into rows and columns of a Pandas DataFrame
    '''
    
    columnNames = []
    
    if isinstance(columns, collections.Mapping):              
        for column in columns:
            columnNames.append(columns[column])
    else:
        columnNames = columns
                
    print(columnNames)
    
    # TODO: Detect column names automatically, remove namespace from them for df.
    out_df = pd.DataFrame(columns=columnNames)
    for node in xmlRoot: 
        values = []
        for column in columns:
            cell = node.find(namespace + column)
            val = cell.text if cell is not None else None
            values.append(val)

        out_df = out_df.append(
            pd.Series(values, index=columnNames), 
            ignore_index = True
        )
        
    return out_df
