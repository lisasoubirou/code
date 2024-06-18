import numpy as np

# Used for the table pretty printing
from engineering_notation import EngNumber # type: ignore

from rcsparameters.chain.chain import RCSChain

# Helper function to print the RCS parameters in a table that can be directly pasted in a LaTeX document
def latex_print_acceleration_attributes(rcs_chain: RCSChain,
                                        rcs_index_to_print: list,
                                        rcs_index_start: int=0,
                                        table_caption: str='General properties of the RCS chain',
                                        table_label: str='tab:general_properties_rcs_chain',
                                        suppress_output: bool=True,
                                        ) -> str:
    '''
    A function to print a LaTeX table of the general properties (Injection/ejection energy, survival rate...)
    of each RCS in a given RCSChain.

    rcs_chain: the RCSChain object of which we want to print the properties
    rcs_index_to_print: list with the index of the RCS we want to 
    rcs_index_start: the RCS number at which the given rcs_chain starts. Default to 1
    table_caption: the table caption
    table_label: the table label used to reference it in the LaTeX document

    return: a string with the full LaTeX code of the table
    '''

    attributes_to_print_acceleration = ['injection_energy',
                                        'ejection_energy',
                                        'energy_ratio',
                                        'injection_gamma',
                                        'ejection_gamma',
                                        'survival_rate',
                                        'acceleration_time',
                                        'linear_acceleration_gradient_for_survival',
                                        'ramping_rate',
                                        'radius',
                                        'circumference',
                                        'packing_fraction',
                                        'bending_radius',
                                        'L_NC', 'L_SC',
                                        'B_SC_max',
                                        'B_injection_average', 'B_ejection_average',
                                        'B_NC_injection', 'B_NC_ejection']

    attributes_clear_name_list = [
                            r'Injection energy             & $E_\text{inj}$              & [eV/u]      &',
                            r'Ejection energy              & $E_\text{ej}$               & [eV/u]      &',
                            r'Energy ratio                 & $E_\text{ej}/E_\text{inj}$  & [-]         &',
                            r'Injection Lorentz factor     & $\gamma_\text{inj}$         & [-]         &',
                            r'Ejection Lorentz factor      & $\gamma_\text{ej}$          & [-]         &',
                            r'Survival rate                & $N_\text{ej}/N_\text{inj}$  & [-]         &',
                            r'Acceleration time            & $\tau_\text{acc}$           & [s]         &',
                            r'Average accel. gradient      & $G_\text{avg}$              & [V/m]       &',
                            r'Ramp rate                    & $\dot{B}_\text{NC}$         & [T/s]       &',
                            r'Machine radius               & $R$                         & [m]         &',
                            r'Circumference                & $2\pi R$                    & [m]         &',
                            r'Pack fraction                & -                           & [-]         &',
                            r'Bend radius                  & $\rho_B$                    & [m]         &',
                            r'Total NC dipole length       & $L_\text{NC}$               & [m]         &',
                            r'Total SC dipole length       & $L_\text{SC}$               & [m]         &',
                            r'SC dipole field              & $B_\text{SC}$               & [T]         &',
                            r'Average Injection dipole field   & $B_\text{inj}$          & [T]         &',
                            r'Average ejection dipole field    & $B_\text{ej}$           & [T]         &',
                            r'Injection NC dipole field        & $B_\text{NC,inj}$       & [T]         &',
                            r'Ejection NC dipole field         & $B_\text{NC,ej}$        & [T]         &',]

    list_of_table_lines = []

    table_columns_alignment = ' c' * rcs_chain.number_of_stages
    table_column_header = ' & '.join([f'RCS {ii}' for ii in np.arange(rcs_index_start,
                                                                      rcs_chain.number_of_stages+rcs_index_start)])

    # The table header, assuming we use booktabs package in LaTeX
    table_header_string = f'''
\\begin{{table}}[htbp]
    \\centering
    \\caption{{{table_caption}}}
    \\label{{{table_label}}}
    \\begin{{tabular}}{{{table_columns_alignment}}}
    \\\toprule
    Parameter & Symbol & Unit & {table_column_header} \\\\
    \\midrule'''

    list_of_table_lines.append(table_header_string)

    # Construct the table rows. Use one column per RCS
    # We also print in the console output each line for a quick view of the results
    for ii_attribute, attributes_clear_name in enumerate(attributes_clear_name_list):

        string_rcs_values = ''

        attribute = attributes_to_print_acceleration[ii_attribute]

        for ii_rcs_to_show in rcs_index_to_print:
            if attribute not in ['survival_rate', 'packing_fraction']:
                # EngNumber is used since in general each attributes has different order of magnitudes
                # and we can not use the SI prefixes
                value_to_print = EngNumber(getattr(rcs_chain.list_of_rcs[ii_rcs_to_show], attribute),
                                           precision=3)
                string_rcs_values += f'{value_to_print}   &'
            else:
                string_rcs_values += f'{getattr(rcs_chain.list_of_rcs[ii_rcs_to_show], attribute):.2f}   &'


        string_to_print = attributes_clear_name + string_rcs_values[:-1] + '\\\\'
        if not suppress_output:
            print(string_to_print)
        list_of_table_lines.append('    ' + string_to_print + '\n')

    # The table footer
    table_footer = '''
\\bottomrule
\\end{tabular}
\\end{table}'''
    list_of_table_lines.append(table_footer)

    return ''.join(list_of_table_lines)


# Helper function to print the RCS RF parameters in a table that can be directly pasted in a LaTeX document
def latex_print_rf_attributes(rcs_chain: RCSChain,
                              rcs_index_to_print: list,
                              rcs_index_start: int=1,
                              table_caption: str='RF properties of the RCS chain',
                              table_label: str='tab:rf_properties_rcs_chain',
                              suppress_output: bool=True,
                              ) -> str:
    '''
    A function to print a LaTeX table of the RF attributes of each RCS in a given RCSChain

    rcs_chain: the RCSChain object of which we want to print the properties
    rcs_index_start: the RCS number at which the given rcs_chain starts. Default to 1
    table_caption: the table caption
    table_label: the table label used to reference it in the LaTeX document

    return: a string with the full LaTeX code of the table
    '''

    # The attributes of the RF class that will be printed
    # Some are deactivated -> they should be added to the attributes_clear_name_list table
    attributes_to_print_rf = [
    'acceleration_time',
    'number_of_cavities',
    'number_of_klystrons',
    'number_of_turns',
    'average_wall_power_consumption',
    'total_wall_power_consumption',
    ]

    # This is used to generate the pretty LaTeX tables
    attributes_clear_name_list = [r'Acceleration time        & $\tau_\text{acc}$          & [s]     &',
                                  r'Number of cavities       & $N_\text{cavities}$         & [-]     &',
                                  r'Number of klystrons      & $N_\text{klystrons}$        & [-]     &',
                                  r'Number of turns          & $N_\text{turns}$            & [-]     &',
                                  r'Average power            & $P_\text{wall, average}$    & [W]     &',
                                  r'Peak all power           & $P_\text{wall, total}$      & [W]     &',
                                      ]

    list_of_table_lines = []

    table_columns_alignment = ' c' * rcs_chain.number_of_stages
    table_column_header = ' & '.join([f'RCS {ii}' for ii in np.arange(rcs_index_start,
                                                                      rcs_chain.number_of_stages+rcs_index_start)])

    # The table header, assuming we use booktabs package in LaTeX
    table_header_string = f'''
\\begin{{table}}[htbp]
    \\centering
    \\caption{{{table_caption}}}
    \\label{{{table_label}}}
    \\begin{{tabular}}{{{table_columns_alignment}}}
    \\toprule
    Parameter & Symbol & Unit & {table_column_header} \\\\
    \\midrule'''

    list_of_table_lines.append(table_header_string)

    # Construct the table rows. Use one column per RCS
    # We also print in the console output each line for a quick view of the results
    for ii_attribute, attributes_clear_name in enumerate(attributes_clear_name_list):

        string_rcs_values = ''

        attribute = attributes_to_print_rf[ii_attribute]

        for ii_rcs_to_show in rcs_index_to_print:
            if attribute not in ['actual_survival_rate', 'packing_fraction']:
                # EngNumber is used since in general each attribute has different order of
                # magnitudes and we can not use the SI prefixes
                value_to_print = EngNumber(getattr(rcs_chain.list_of_rcs[ii_rcs_to_show].rf, attribute),
                                           precision=3)
                string_rcs_values += f'{value_to_print}   &'
            else:
                string_rcs_values += f'{getattr(rcs_chain.list_of_rcs[ii_rcs_to_show].rf, attribute):.2f}   &'


        string_to_print = attributes_clear_name + string_rcs_values[:-1] + '\\\\'
        if not suppress_output:
            print(string_to_print)
        list_of_table_lines.append('    ' + string_to_print + '\n')

    # The table footer
    table_footer = '''
\\bottomrule
\\end{tabular}
\\end{table}'''
    list_of_table_lines.append(table_footer)

    return ''.join(list_of_table_lines)
